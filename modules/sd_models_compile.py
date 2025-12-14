import time
import logging
import torch
from sdnext_core import MODELDATA
from modules import shared, devices, sd_models, errors
from installer import setup_logging


#Used by OpenVINO, can be used with TensorRT or Olive
class CompiledModelState:
    def __init__(self):
        self.is_compiled = False
        self.model_hash_str = ""
        self.first_pass = True
        self.first_pass_refiner = True
        self.first_pass_vae = True
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.partition_id = 0
        self.cn_model = []
        self.lora_model = []
        self.compiled_cache = {}
        self.req_cache = {}
        self.partitioned_modules = {}


deepcache_worker = None


def ipex_optimize(sd_model, apply_to_components=True, op="Model"):
    try:
        t0 = time.time()
        import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import

        def ipex_optimize_model(model, op=None, sd_model=None): # pylint: disable=unused-argument
            model.eval()
            model.training = False
            if model.device.type != "meta":
                return_device = model.device
                model = ipex.optimize(model.to(devices.device),
                    dtype=devices.dtype,
                    inplace=True,
                    weights_prepack=False
                ).to(return_device) # pylint: disable=attribute-defined-outside-init
            else:
                model = ipex.optimize(model,
                    dtype=devices.dtype,
                    inplace=True,
                    weights_prepack=False
                ) # pylint: disable=attribute-defined-outside-init
            devices.torch_gc()
            return model

        if apply_to_components:
            sd_model = sd_models.apply_function_to_model(sd_model, ipex_optimize_model, shared.opts.ipex_optimize, op="ipex")
        else:
            sd_model = ipex_optimize_model(sd_model, op=op)

        t1 = time.time()
        shared.log.info(f"{op} IPEX Optimize: time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"{op} IPEX Optimize: error: {e}")
    return sd_model


def optimize_openvino(sd_model, clear_cache=True):
    try:
        from modules.intel.openvino import openvino_fx # pylint: disable=unused-import
        if clear_cache and shared.compiled_model_state is not None:
            shared.compiled_model_state.compiled_cache.clear()
            shared.compiled_model_state.req_cache.clear()
            shared.compiled_model_state.partitioned_modules.clear()
        if clear_cache or shared.compiled_model_state is None:
            shared.compiled_model_state = CompiledModelState()
            shared.compiled_model_state.is_compiled = True
            shared.compiled_model_state.first_pass = 'precompile' not in shared.opts.cuda_compile_options
            shared.compiled_model_state.first_pass_vae = 'precompile' not in shared.opts.cuda_compile_options
            shared.compiled_model_state.first_pass_refiner = 'precompile' not in shared.opts.cuda_compile_options
        sd_models.set_accelerate(sd_model)
    except Exception as e:
        shared.log.warning(f"Model compile: task=OpenVINO: {e}")
    return sd_model


def compile_onediff(sd_model):
    try:
        from onediff.infer_compiler import oneflow_compile

    except Exception as e:
        shared.log.warning(f"Model compile: task=onediff {e}")
        return sd_model

    try:
        t0 = time.time()
        # For some reason compiling the text_encoder, when it is used by
        # the 'compel' package which sdnext uses, it becomes 100 times
        # slower as if it is recompiling every time.
        #sd_model.text_encoder = oneflow_compile(sd_model.text_encoder)
        #if hasattr(sd_model, 'text_endcoder_2'):
        #    sd_model.text_encoder_2 = oneflow_compile(sd_model.text_encoder_2)
        sd_model.unet = oneflow_compile(sd_model.unet)
        sd_model.vae.encoder = oneflow_compile(sd_model.vae.encoder)
        sd_model.vae.decoder = oneflow_compile(sd_model.vae.decoder)
        # How are Loras, Adaptors, and other things compiled

        # DW: I'm unclear whether this is also a problem with onediff
        # as it was for sfast.
        setup_logging() # compile messes with logging so reset is needed
        if 'precompile' in shared.opts.cuda_compile_options:
            shared.log.debug("Model compile: task=onediff precompile")
            sd_model("dummy prompt")
        t1 = time.time()
        shared.log.info(f"Model compile: task=onediff time={t1-t0:.2f}")
    except Exception as e:
        shared.log.info(f"Model compile: task=onediff {e}")
    return sd_model


def compile_stablefast(sd_model):
    try:
        import sfast.compilers.stable_diffusion_pipeline_compiler as sf
    except Exception as e:
        shared.log.warning(f'Model compile: task=stablefast: {e}')
        return sd_model
    config = sf.CompilationConfig.Default()
    try:
        import xformers # pylint: disable=unused-import
        config.enable_xformers = True
    except Exception:
        pass
    try:
        import triton # pylint: disable=unused-import
        config.enable_triton = True
    except Exception:
        pass
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    config.enable_cuda_graph = 'fullgraph' in shared.opts.cuda_compile_options
    config.enable_jit_freeze = shared.opts.diffusers_eval
    config.memory_format = torch.channels_last if shared.opts.opt_channelslast else torch.contiguous_format
    # config.trace_scheduler = False
    # config.enable_cnn_optimization
    # config.prefer_lowp_gemm
    try:
        t0 = time.time()
        sd_model = sf.compile(sd_model, config)
        sd_model.sfast = True
        setup_logging() # compile messes with logging so reset is needed
        if 'precompile' in shared.opts.cuda_compile_options:
            shared.log.debug("Model compile: task=stablefast precompile")
            sd_model("dummy prompt")
        t1 = time.time()
        shared.log.info(f"Model compile: task=stablefast config={config.__dict__} time={t1-t0:.2f}")
    except Exception as e:
        shared.log.info(f"Model compile: task=stablefast {e}")
    return sd_model


def compile_torch(sd_model, apply_to_components=True, op="Model"):
    try:
        t0 = time.time()
        import torch._dynamo # pylint: disable=unused-import,redefined-outer-name
        torch._dynamo.reset() # pylint: disable=protected-access
        shared.log.debug(f"{op} compile: task=torch backends={torch._dynamo.list_backends()}") # pylint: disable=protected-access

        def torch_compile_model(model, op=None, sd_model=None): # pylint: disable=unused-argument
            if hasattr(model, 'compile_repeated_blocks') and 'repeated' in shared.opts.cuda_compile_options:
                model.compile_repeated_blocks(
                    mode=shared.opts.cuda_compile_mode,
                    backend=shared.opts.cuda_compile_backend,
                    fullgraph='fullgraph' in shared.opts.cuda_compile_options,
                    dynamic='dynamic' in shared.opts.cuda_compile_options,
                )
            elif hasattr(model, 'device') and model.device.type != "meta":
                return_device = model.device
                model = torch.compile(model.to(devices.device),
                    mode=shared.opts.cuda_compile_mode,
                    backend=shared.opts.cuda_compile_backend,
                    fullgraph='fullgraph' in shared.opts.cuda_compile_options,
                    dynamic='dynamic' in shared.opts.cuda_compile_options,
                ).to(return_device)
            else:
                model = torch.compile(model,
                    mode=shared.opts.cuda_compile_mode,
                    backend=shared.opts.cuda_compile_backend,
                    fullgraph='fullgraph' in shared.opts.cuda_compile_options,
                    dynamic='dynamic' in shared.opts.cuda_compile_options,
                )
            devices.torch_gc()
            return model

        if shared.opts.cuda_compile_backend == "openvino_fx":
            sd_model = optimize_openvino(sd_model, clear_cache=apply_to_components)
        elif shared.opts.cuda_compile_backend == "olive-ai":
            if shared.compiled_model_state is None:
                shared.compiled_model_state = CompiledModelState()
            return sd_model
        elif shared.opts.cuda_compile_backend ==  "migraphx":
            import torch_migraphx # pylint: disable=unused-import
        log_level = logging.WARNING if 'verbose' in shared.opts.cuda_compile_options else logging.CRITICAL # pylint: disable=protected-access
        if hasattr(torch, '_logging'):
            torch._logging.set_logs(dynamo=log_level, aot=log_level, inductor=log_level) # pylint: disable=protected-access
        torch._dynamo.config.verbose = 'verbose' in shared.opts.cuda_compile_options # pylint: disable=protected-access
        torch._dynamo.config.suppress_errors = 'verbose' not in shared.opts.cuda_compile_options # pylint: disable=protected-access

        try:
            torch._inductor.config.conv_1x1_as_mm = True # pylint: disable=protected-access
            torch._inductor.config.coordinate_descent_tuning = True # pylint: disable=protected-access
            torch._inductor.config.epilogue_fusion = False # pylint: disable=protected-access
            torch._inductor.config.coordinate_descent_check_all_directions = True # pylint: disable=protected-access
            torch._inductor.config.use_mixed_mm = True # pylint: disable=protected-access
            # torch._inductor.config.force_fuse_int_mm_with_mul = True # pylint: disable=protected-access
        except Exception as e:
            shared.log.error(f"{op} compile: torch inductor config error: {e}")

        if apply_to_components:
            sd_model = sd_models.apply_function_to_model(sd_model, function=torch_compile_model, options=shared.opts.cuda_compile, op="compile")
        else:
            sd_model = torch_compile_model(sd_model)

        setup_logging() # compile messes with logging so reset is needed
        if apply_to_components and 'precompile' in shared.opts.cuda_compile_options:
            try:
                shared.log.debug(f"{op} compile: task=torch precompile")
                sd_model("dummy prompt")
            except Exception:
                pass
        t1 = time.time()
        shared.log.info(f"{op} compile: task=torch time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"{op} compile: task=torch {e}")
        errors.display(e, 'Compile')
    return sd_model


def check_deepcache(enable: bool):
    if deepcache_worker is not None:
        if enable:
            deepcache_worker.enable()
        else:
            deepcache_worker.disable()


def compile_deepcache(sd_model):
    global deepcache_worker # pylint: disable=global-statement
    if not hasattr(sd_model, 'unet'):
        shared.log.warning(f'Model compile: task=deepcache pipeline={sd_model.__class__} not supported')
        return sd_model
    try:
        from DeepCache import DeepCacheSDHelper
    except Exception as e:
        shared.log.warning(f'Model compile: task=deepcache {e}')
        return sd_model
    t0 = time.time()
    check_deepcache(False)
    deepcache_worker = DeepCacheSDHelper(pipe=sd_model)
    deepcache_worker.set_params(cache_interval=shared.opts.deep_cache_interval, cache_branch_id=0)
    t1 = time.time()
    shared.log.info(f"Model compile: task=deepcache config={deepcache_worker.params} time={t1-t0:.2f}")
    # config={'cache_interval': 3, 'cache_layer_id': 0, 'cache_block_id': 0, 'skip_mode': 'uniform'} time=0.00
    return sd_model


def compile_diffusers(sd_model, apply_to_components=True, op="Model"):
    if shared.opts.cuda_compile_backend == 'none':
        shared.log.warning(f'{op} compile enabled but no backend specified')
        return sd_model
    shared.log.info(f"{op} compile: pipeline={sd_model.__class__.__name__} mode={shared.opts.cuda_compile_mode} backend={shared.opts.cuda_compile_backend} options={shared.opts.cuda_compile_options} compile={shared.opts.cuda_compile}")
    if shared.opts.cuda_compile_backend == 'onediff':
        sd_model = compile_onediff(sd_model)
    elif shared.opts.cuda_compile_backend == 'stable-fast':
        sd_model = compile_stablefast(sd_model)
    elif shared.opts.cuda_compile_backend == 'deep-cache':
        sd_model = compile_deepcache(sd_model)
    else:
        check_deepcache(False)
        sd_model = compile_torch(sd_model, apply_to_components=apply_to_components, op=op)
    return sd_model


def openvino_recompile_model(p, hires=False, refiner=False): # recompile if a parameter changes # pylint: disable=unused-argument
    if shared.opts.cuda_compile_backend == "openvino_fx" and 'Model' in shared.opts.cuda_compile:
        compile_height = p.height if not hires and hasattr(p, 'height') else p.hr_upscale_to_y
        compile_width = p.width if not hires and hasattr(p, 'width') else p.hr_upscale_to_x
        """
        if shared.compiled_model_state is None:
            openvino_first_pass = True
        else:
            if refiner:
                openvino_first_pass = shared.compiled_model_state.first_pass_refiner
            else:
                openvino_first_pass = shared.compiled_model_state.first_pass
        if (shared.compiled_model_state is None or
            (
            not openvino_first_pass
            and (
                    shared.compiled_model_state.height != compile_height
                    or shared.compiled_model_state.width != compile_width
                    or shared.compiled_model_state.batch_size != p.batch_size
                )
            )):
            if refiner:
                shared.log.info("OpenVINO: Recompiling refiner")
                sd_models.unload_model_weights(op='refiner')
                sd_models.reload_model_weights(op='refiner')
            else:
                shared.log.info("OpenVINO: Recompiling base model")
                sd_models.unload_model_weights(op='model')
                sd_models.reload_model_weights(op='model')
        """
        if shared.compiled_model_state is None:
            shared.log.warning("OpenVINO: Compile Model State is not found, model is not compiled!")
        else:
            shared.compiled_model_state.height = compile_height
            shared.compiled_model_state.width = compile_width
            shared.compiled_model_state.batch_size = p.batch_size


def openvino_post_compile(op="base"): # delete unet after OpenVINO compile
    if shared.opts.cuda_compile_backend == "openvino_fx" and 'Model' in shared.opts.cuda_compile:
        if shared.compiled_model_state.first_pass and op == "base":
            shared.compiled_model_state.first_pass = False
            if not shared.opts.openvino_disable_memory_cleanup and hasattr(MODELDATA.sd_model, "unet"):
                MODELDATA.sd_model.unet.apply(sd_models.convert_to_faketensors)
                devices.torch_gc(force=True)
        if shared.compiled_model_state.first_pass_refiner and op == "refiner":
            shared.compiled_model_state.first_pass_refiner = False
            if not shared.opts.openvino_disable_memory_cleanup and hasattr(MODELDATA.sd_refiner, "unet"):
                MODELDATA.sd_refiner.unet.apply(sd_models.convert_to_faketensors)
                devices.torch_gc(force=True)
