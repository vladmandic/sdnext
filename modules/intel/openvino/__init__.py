import os
import sys
import torch
import nncf

from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino.frontend import FrontEndManager
from openvino import Core, Type, PartialShape, serialize
from openvino.properties import hint as ov_hints

from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._dynamo.backends.registry import register_backend
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten

from types import MappingProxyType
from hashlib import sha256
import functools

from modules import shared, devices, sd_models


# importing openvino.runtime forces DeprecationWarning to "always"
# And Intel's own libs (NNCF) imports the deprecated module
# Reset the warnings back to ignore:
try:
    import warnings
    import openvino.runtime # pylint: disable=unused-import
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)
except Exception:
    pass


# Set default params
torch._dynamo.config.cache_size_limit = 64 # pylint: disable=protected-access
torch._dynamo.eval_frame.check_if_dynamo_supported = lambda: True # pylint: disable=protected-access
if hasattr(torch._dynamo.config, "inline_inbuilt_nn_modules"):
    torch._dynamo.config.inline_inbuilt_nn_modules = False # pylint: disable=protected-access


DEFAULT_OPENVINO_PYTHON_CONFIG = MappingProxyType(
    {
        "use_python_fusion_cache": True,
        "allow_single_op_fusion": True,
    },
)


class OpenVINOGraphModule(torch.nn.Module):
    def __init__(self, gm, partition_id, use_python_fusion_cache, model_hash_str: str = None, file_name="", int_inputs=[]):
        super().__init__()
        self.gm = gm
        self.int_inputs = int_inputs
        self.partition_id = partition_id
        self.executor_parameters = {"use_python_fusion_cache": use_python_fusion_cache,
                                    "model_hash_str": model_hash_str}
        self.file_name = file_name

    def __call__(self, *args):
        ov_inputs = []
        for arg in args:
            if not isinstance(arg, int):
                ov_inputs.append(arg)
        for idx, int_input in self.int_inputs:
            ov_inputs.insert(idx, int_input)
        result = openvino_execute(self.gm, *ov_inputs, executor_parameters=self.executor_parameters, partition_id=self.partition_id, file_name=self.file_name)
        return result


def get_device_list():
    core = Core()
    return core.available_devices


def get_device():
    if hasattr(shared, "opts") and len(shared.opts.openvino_devices) == 1:
        return shared.opts.openvino_devices[0]

    core = Core()
    if hasattr(shared, "opts") and len(shared.opts.openvino_devices) > 1:
        device = ""
        available_devices = shared.opts.openvino_devices.copy()
        if "CPU" in shared.opts.openvino_devices:
            available_devices.remove("CPU")
        for hetero_device in available_devices:
            device = f"{device},{hetero_device}"
        if "CPU" in shared.opts.openvino_devices:
            device = f"{device},CPU"
        device = f"HETERO:{device[1:]}"
    elif any(openvino_cpu in cpu_module.lower() for cpu_module in shared.cmd_opts.use_cpu for openvino_cpu in ["openvino", "all"]):
        device = "CPU"
    elif shared.cmd_opts.device_id is not None:
        device = f"GPU.{shared.cmd_opts.device_id}"
        if device not in core.available_devices:
            device = "GPU.0" if "GPU.0" in core.available_devices else "GPU" if "GPU" in core.available_devices else "CPU"
    elif "GPU" in core.available_devices:
        device = "GPU"
    elif "GPU.1" in core.available_devices:
        device = "GPU.1"
    elif "GPU.0" in core.available_devices:
        device = "GPU.0"
    else:
        device = core.available_devices[-1]
        shared.log.warning(f"OpenVINO: No compatible GPU detected! Using {device}")
    return device


def get_openvino_device():
    core = Core()
    try:
        return core.get_property(get_device(), "FULL_DEVICE_NAME")
    except Exception:
        return f"OpenVINO {get_device()}"


def cached_model_name(model_hash_str, device, args, cache_root, reversed = False):
    if model_hash_str is None:
        return None

    model_cache_dir = cache_root + "/model/"

    try:
        os.makedirs(model_cache_dir, exist_ok=True)
        file_name = model_cache_dir + model_hash_str + "_" + device
    except OSError as error:
        shared.log.error(f"Cache directory {cache_root} cannot be created. Model caching is disabled. Error: {error}")
        return None

    inputs_str = ""
    for input_data in args:
        if isinstance(input_data, torch.SymInt):
            if reversed:
                inputs_str = "_" + "torch.SymInt[]" + inputs_str
            else:
                inputs_str += "_" + "torch.SymInt[]"
        elif isinstance(input_data, int):
            pass
        else:
            if reversed:
                inputs_str = "_" + str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "") + inputs_str
            else:
                inputs_str += "_" + str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "")
    inputs_str = sha256(inputs_str.encode('utf-8')).hexdigest()
    file_name += "_" + inputs_str

    return file_name


def execute(
    gm,
    *args,
    executor = "openvino",
    executor_parameters = None,
    file_name = ""
):
    if executor == "openvino":
        return openvino_execute_partitioned(gm, *args, executor_parameters=executor_parameters, file_name=file_name)
    elif executor == "strictly_openvino":
        return openvino_execute(gm, *args, executor_parameters=executor_parameters, file_name=file_name)

    msg = "Received unexpected value for 'executor': {0}. Allowed values are: openvino, strictly_openvino.".format(executor)
    raise ValueError(msg)


def execute_cached(compiled_model, *args):
    flat_args, _ = tree_flatten(args)
    ov_inputs = [a.detach().cpu().numpy() for a in flat_args]

    if (shared.compiled_model_state.cn_model == []):
        ov_inputs.reverse()

    res = compiled_model(ov_inputs)
    result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
    return result

def openvino_compile(gm: GraphModule, *example_inputs, model_hash_str: str = None, file_name=""):
    core = Core()

    device = get_device()
    global dont_use_4bit_nncf
    global dont_use_nncf
    global dont_use_quant

    if file_name is not None and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin"):
        om = core.read_model(file_name + ".xml")
    else:
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework("pytorch")

        input_shapes = []
        input_types = []
        for input_data in example_inputs:
            if isinstance(input_data, torch.SymInt):
                input_types.append(torch.SymInt)
                input_shapes.append(torch.Size([]))
            elif isinstance(input_data, int):
                pass
            else:
                input_types.append(input_data.type())
                input_shapes.append(input_data.size())

        decoder = TorchFXPythonDecoder(gm, input_shapes=input_shapes, input_types=input_types)
        im = fe.load(decoder)
        om = fe.convert(im)

        if file_name is not None:
            serialize(om, file_name + ".xml", file_name + ".bin")
            if (shared.compiled_model_state.cn_model != []):
                f = open(file_name + ".txt", "w")
                for input_data in example_inputs:
                    f.write(str(input_data.size()))
                    f.write("\n")
                f.close()

    dtype_mapping = {
        torch.float32: Type.f32,
        torch.float64: Type.f64,
        torch.float16: Type.f16,
        torch.int64: Type.i64,
        torch.int32: Type.i32,
        torch.uint8: Type.u8,
        torch.int8: Type.i8,
        torch.bool: Type.boolean
    }

    idx_minus = 0
    for idx, input_data in enumerate(example_inputs):
        if isinstance(input_data, int):
            idx_minus += 1
        else:
            om.inputs[idx-idx_minus].get_node().set_element_type(dtype_mapping[input_data.dtype])
            om.inputs[idx-idx_minus].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
    om.validate_nodes_and_infer_types()

    if shared.opts.nncf_quantize and not dont_use_quant:
        new_inputs = []
        for idx, _ in enumerate(example_inputs):
            new_inputs.append(example_inputs[idx].detach().cpu().numpy())
        new_inputs = [new_inputs]
        if shared.opts.nncf_quantize_mode == "INT8":
            om = nncf.quantize(om, nncf.Dataset(new_inputs))
        else:
            om = nncf.quantize(om, nncf.Dataset(new_inputs), mode=getattr(nncf.QuantizationMode, shared.opts.nncf_quantize_mode),
                advanced_parameters=nncf.quantization.advanced_parameters.AdvancedQuantizationParameters(
                overflow_fix=nncf.quantization.advanced_parameters.OverflowFix.DISABLE, backend_params=None))

    if shared.opts.nncf_compress_weights and not dont_use_nncf:
        if dont_use_4bit_nncf or shared.opts.nncf_compress_weights_mode == "INT8":
            om = nncf.compress_weights(om)
        else:
            compress_group_size = shared.opts.nncf_compress_weights_group_size if shared.opts.nncf_compress_weights_group_size != 0 else None
            compress_ratio = shared.opts.nncf_compress_weights_raito if shared.opts.nncf_compress_weights_raito != 0 else None
            om = nncf.compress_weights(om, mode=getattr(nncf.CompressWeightsMode, shared.opts.nncf_compress_weights_mode), group_size=compress_group_size, ratio=compress_ratio)

    hints = {}
    if shared.opts.openvino_accuracy == "performance":
        hints[ov_hints.execution_mode] = ov_hints.ExecutionMode.PERFORMANCE
    elif shared.opts.openvino_accuracy == "accuracy":
        hints[ov_hints.execution_mode] = ov_hints.ExecutionMode.ACCURACY
    if model_hash_str is not None:
        hints['CACHE_DIR'] = shared.opts.openvino_cache_path + '/blob'
    core.set_property(hints)
    dont_use_nncf = False
    dont_use_quant = False
    dont_use_4bit_nncf = False

    compiled_model = core.compile_model(om, device)
    return compiled_model


def openvino_compile_cached_model(cached_model_path, *example_inputs):
    core = Core()
    om = core.read_model(cached_model_path + ".xml")

    global dont_use_4bit_nncf
    global dont_use_nncf
    global dont_use_quant

    dtype_mapping = {
        torch.float32: Type.f32,
        torch.float64: Type.f64,
        torch.float16: Type.f16,
        torch.int64: Type.i64,
        torch.int32: Type.i32,
        torch.uint8: Type.u8,
        torch.int8: Type.i8,
        torch.bool: Type.boolean
    }

    for idx, input_data in enumerate(example_inputs):
        om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
        om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
    om.validate_nodes_and_infer_types()

    if shared.opts.nncf_quantize and not dont_use_quant:
        new_inputs = []
        for idx, _ in enumerate(example_inputs):
            new_inputs.append(example_inputs[idx].detach().cpu().numpy())
        new_inputs = [new_inputs]
        if shared.opts.nncf_quantize_mode == "INT8":
            om = nncf.quantize(om, nncf.Dataset(new_inputs))
        else:
            om = nncf.quantize(om, nncf.Dataset(new_inputs), mode=getattr(nncf.QuantizationMode, shared.opts.nncf_quantize_mode),
                advanced_parameters=nncf.quantization.advanced_parameters.AdvancedQuantizationParameters(
                overflow_fix=nncf.quantization.advanced_parameters.OverflowFix.DISABLE, backend_params=None))

    if shared.opts.nncf_compress_weights and not dont_use_nncf:
        if dont_use_4bit_nncf or shared.opts.nncf_compress_weights_mode == "INT8":
            om = nncf.compress_weights(om)
        else:
            compress_group_size = shared.opts.nncf_compress_weights_group_size if shared.opts.nncf_compress_weights_group_size != 0 else None
            compress_ratio = shared.opts.nncf_compress_weights_raito if shared.opts.nncf_compress_weights_raito != 0 else None
            om = nncf.compress_weights(om, mode=getattr(nncf.CompressWeightsMode, shared.opts.nncf_compress_weights_mode), group_size=compress_group_size, ratio=compress_ratio)

    hints = {'CACHE_DIR': shared.opts.openvino_cache_path + '/blob'}
    if shared.opts.openvino_accuracy == "performance":
        hints[ov_hints.execution_mode] = ov_hints.ExecutionMode.PERFORMANCE
    elif shared.opts.openvino_accuracy == "accuracy":
        hints[ov_hints.execution_mode] = ov_hints.ExecutionMode.ACCURACY
    core.set_property(hints)
    dont_use_nncf = False
    dont_use_quant = False
    dont_use_4bit_nncf = False

    compiled_model = core.compile_model(om, get_device())
    return compiled_model


def openvino_execute(gm: GraphModule, *args, executor_parameters=None, partition_id=None, file_name=""):
    if hasattr(gm, "partition_id"):
        partition_id = gm.partition_id
    if hasattr(gm, "gm"):
        gm = gm.gm
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    use_cache = partition_id is not None and executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )

    model_hash_str = executor_parameters.get("model_hash_str", None)
    if model_hash_str is not None:
        model_hash_str = model_hash_str + str(partition_id) if partition_id is not None else ""

    if use_cache and (partition_id in shared.compiled_model_state.compiled_cache.keys()):
        compiled = shared.compiled_model_state.compiled_cache[partition_id]
        req = shared.compiled_model_state.req_cache[partition_id]
    else:
        if (shared.compiled_model_state.cn_model != [] and file_name is not None
                and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin")):
            compiled = openvino_compile_cached_model(file_name, *args)
        else:
            compiled = openvino_compile(gm, *args, model_hash_str=model_hash_str, file_name=file_name)
        if use_cache:
            shared.compiled_model_state.compiled_cache[partition_id] = compiled
        req = compiled.create_infer_request()
        if use_cache:
            shared.compiled_model_state.req_cache[partition_id] = req

    flat_args, _ = tree_flatten(args)
    ov_inputs = []
    for arg in flat_args:
        if not isinstance(arg, int):
            ov_inputs.append((arg.detach().cpu().numpy()))

    res = req.infer(ov_inputs, share_inputs=True, share_outputs=True)

    results1 = [torch.from_numpy(res[out]) for out in compiled.outputs]
    if len(results1) == 1:
        return results1[0]
    return results1


def openvino_execute_partitioned(gm: GraphModule, *args, executor_parameters=None, file_name=""):
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    use_python_fusion_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    model_hash_str = executor_parameters.get("model_hash_str", None)

    if file_name:
        signature = file_name.rsplit("/", maxsplit=1)[-1].split("_fs", maxsplit=1)[0]
    else:
        signature = "signature"
    if model_hash_str is None:
        file_name = None

    idx_minus = 0
    int_inputs = []
    for idx, input_data in enumerate(args):
        if isinstance(input_data, int):
            int_inputs.append([idx, input_data])
            idx_minus += 1
        elif isinstance(input_data, torch.Tensor):
            signature = signature + "_" + str(idx-idx_minus) + ":" + str(input_data.type())[6:] + ":" + str(input_data.size())[11:-1].replace(" ", "")
        else:
            signature = signature + "_" + str(idx-idx_minus) + ":" + type(input_data).__name__ + ":val(" + str(input_data) + ")"

    if signature not in shared.compiled_model_state.partitioned_modules:
        shared.compiled_model_state.partitioned_modules[signature] = partition_graph(gm,  use_python_fusion_cache=use_python_fusion_cache,
                                                        model_hash_str=model_hash_str, file_name=file_name, int_inputs=int_inputs)

    ov_inputs = []
    for arg in args:
        if not isinstance(arg, int):
            ov_inputs.append(arg)
    for idx, int_input in shared.compiled_model_state.partitioned_modules[signature][1]:
        ov_inputs.insert(idx, int_input)
    return shared.compiled_model_state.partitioned_modules[signature][0](*ov_inputs)


def partition_graph(gm: GraphModule, use_python_fusion_cache: bool, model_hash_str: str = None, file_name="", int_inputs=[]):
    for node in gm.graph.nodes:
        if node.op == "call_module" and "fused_" in node.name:
            openvino_submodule = getattr(gm, node.name)
            if isinstance(openvino_submodule, OpenVINOGraphModule):
                int_inputs = openvino_submodule.int_inputs
                continue
            gm.delete_submodule(node.target)
            gm.add_submodule(
                node.target,
                OpenVINOGraphModule(
                    openvino_submodule, shared.compiled_model_state.partition_id, use_python_fusion_cache,
                    model_hash_str=model_hash_str, file_name=file_name, int_inputs=int_inputs),
            )
            shared.compiled_model_state.partition_id += 1

    return gm, int_inputs


def generate_subgraph_str(tensor):
    if hasattr(tensor, "weight"):
        shared.compiled_model_state.model_hash_str = shared.compiled_model_state.model_hash_str + sha256(str(tensor.weight).encode('utf-8')).hexdigest()
    return tensor


def get_subgraph_type(tensor):
    global subgraph_type
    subgraph_type.append(type(tensor))
    return tensor


@fake_tensor_unsupported
def openvino_fx(subgraph, example_inputs, options=None):
    global dont_use_4bit_nncf
    global dont_use_nncf
    global dont_use_quant
    global subgraph_type

    dont_use_4bit_nncf = False
    dont_use_nncf = False
    dont_use_quant = False
    dont_use_faketensors = False
    executor_parameters = None
    inputs_reversed = False
    maybe_fs_cached_name = None

    subgraph_type = []
    subgraph.apply(get_subgraph_type)

    # SD 1.5 / SDXL VAE
    if (subgraph_type[0] is torch.nn.modules.conv.Conv2d and
        subgraph_type[1] is torch.nn.modules.conv.Conv2d and
        subgraph_type[2] is torch.nn.modules.normalization.GroupNorm and
        subgraph_type[3] is torch.nn.modules.activation.SiLU):

        dont_use_4bit_nncf = True
        dont_use_nncf = bool("VAE" not in shared.opts.nncf_compress_weights)
        dont_use_quant = bool("VAE" not in shared.opts.nncf_quantize)

    # SD 1.5 / SDXL Text Encoder
    elif (subgraph_type[0] is torch.nn.modules.sparse.Embedding and
        subgraph_type[1] is torch.nn.modules.sparse.Embedding and
        subgraph_type[2] is torch.nn.modules.normalization.LayerNorm and
        subgraph_type[3] is torch.nn.modules.linear.Linear):

        dont_use_faketensors = True
        dont_use_nncf = bool("TE" not in shared.opts.nncf_compress_weights)
        dont_use_quant = bool("TE" not in shared.opts.nncf_quantize)

    # Create a hash to be used for caching
    shared.compiled_model_state.model_hash_str = ""
    subgraph.apply(generate_subgraph_str)
    #shared.compiled_model_state.model_hash_str = shared.compiled_model_state.model_hash_str + sha256(subgraph.code.encode('utf-8')).hexdigest()
    shared.compiled_model_state.model_hash_str = sha256(shared.compiled_model_state.model_hash_str.encode('utf-8')).hexdigest()

    # Check if the model was fully supported and already cached
    example_inputs.reverse()
    inputs_reversed = True
    maybe_fs_cached_name = cached_model_name(shared.compiled_model_state.model_hash_str + "_fs", get_device(), example_inputs, shared.opts.openvino_cache_path)
    if not shared.opts.openvino_disable_model_caching:
        os.environ.setdefault('OPENVINO_TORCH_MODEL_CACHING', "1")
        executor_parameters = {"model_hash_str": shared.compiled_model_state.model_hash_str}

        if os.path.isfile(maybe_fs_cached_name + ".xml") and os.path.isfile(maybe_fs_cached_name + ".bin"):
            example_inputs_reordered = []
            if (os.path.isfile(maybe_fs_cached_name + ".txt")):
                f = open(maybe_fs_cached_name + ".txt", "r")
                for input_data in example_inputs:
                    shape = f.readline()
                    if (str(input_data.size()) != shape):
                        for idx1, input_data1 in enumerate(example_inputs):
                            if (str(input_data1.size()).strip() == str(shape).strip()):
                                example_inputs_reordered.append(example_inputs[idx1])
                example_inputs = example_inputs_reordered

            if dont_use_faketensors or shared.opts.openvino_disable_memory_cleanup:
                pass
            else:
                # Delete unused subgraphs
                subgraph = subgraph.apply(sd_models.convert_to_faketensors)
                devices.torch_gc(force=True)

            # Model is fully supported and already cached. Run the cached OV model directly.
            compiled_model = openvino_compile_cached_model(maybe_fs_cached_name, *example_inputs)

            def _call(*args):
                if (shared.compiled_model_state.cn_model != [] and str(shared.compiled_model_state.cn_model) in maybe_fs_cached_name):
                    args_reordered = []
                    if (os.path.isfile(maybe_fs_cached_name + ".txt")):
                        f = open(maybe_fs_cached_name + ".txt", "r")
                        for input_data in args:
                            shape = f.readline()
                            if (str(input_data.size()) != shape):
                                for idx1, input_data1 in enumerate(args):
                                    if (str(input_data1.size()).strip() == str(shape).strip()):
                                        args_reordered.append(args[idx1])
                    args = args_reordered

                res = execute_cached(compiled_model, *args)
                shared.compiled_model_state.partition_id = shared.compiled_model_state.partition_id + 1
                return res
            return _call
    else:
        os.environ.setdefault('OPENVINO_TORCH_MODEL_CACHING', "0")

    if inputs_reversed:
        example_inputs.reverse()
    model = make_fx(subgraph)(*example_inputs)
    for node in model.graph.nodes:
        if node.target == torch.ops.aten.mul_.Tensor:
            node.target = torch.ops.aten.mul.Tensor
        elif node.target == torch.ops.aten._unsafe_index.Tensor:
            node.target = torch.ops.aten.index.Tensor
    with devices.inference_context():
        model.eval()
    partitioner = Partitioner(options=None)
    compiled_model = partitioner.make_partitions(model, options=None)

    if executor_parameters is not None and 'model_hash_str' in executor_parameters:
        # Check if the model is fully supported.
        fully_supported = partitioner.check_fully_supported(compiled_model)
        if fully_supported:
            executor_parameters["model_hash_str"] += "_fs"

    def _call(*args):
        res = execute(compiled_model, *args, executor="openvino", executor_parameters=executor_parameters, file_name=maybe_fs_cached_name)
        return res
    return _call


if "openvino_fx" not in torch.compiler.list_backends():
    register_backend(compiler_fn=openvino_fx, name="openvino_fx")
