import os
import enum
from typing import Union
from collections import namedtuple
from modules import sd_models, hashes, shared


NetworkWeights = namedtuple('NetworkWeights', ['network_key', 'sd_key', 'w', 'sd_module'])
metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}


class SdVersion(enum.Enum):
    Unknown = 1
    SD1 = 2
    SD2 = 3
    SD3 = 3
    SDXL = 4
    SC = 5
    F1 = 6
    HV = 7


class NetworkOnDisk:
    def __init__(self, name, filename):
        self.shorthash = None
        self.hash = None
        self.name = name
        self.filename = filename
        if filename.startswith(shared.cmd_opts.lora_dir):
            self.fullname = os.path.splitext(filename[len(shared.cmd_opts.lora_dir):].strip("/"))[0]
        else:
            self.fullname = name
        self.metadata = {}
        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"
        if self.is_safetensors:
            self.metadata = sd_models.read_metadata_from_safetensors(filename)
        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v
            self.metadata = m
        self.alias = self.metadata.get('ss_output_name', self.name)
        sha256 = hashes.sha256_from_cache(self.filename, "lora/" + self.name) or hashes.sha256_from_cache(self.filename, "lora/" + self.name, use_addnet_hash=True) or self.metadata.get('sshs_model_hash')
        self.set_hash(sha256)
        self.sd_version = self.detect_version()

    def detect_version(self):
        base = str(self.metadata.get('ss_base_model_version', "")).lower()
        arch = str(self.metadata.get('modelspec.architecture', "")).lower()
        if base.startswith("sd_v1"):
            return 'sd1'
        if base.startswith("sdxl"):
            return 'xl'
        if base.startswith("stable_cascade"):
            return 'sc'
        if base.startswith("sd3"):
            return 'sd3'
        if base.startswith("flux"):
            return 'f1'
        if base.startswith("hunyuan_video"):
            return 'hv'

        if arch.startswith("stable-diffusion-v1"):
            return 'sd1'
        if arch.startswith("stable-diffusion-xl"):
            return 'xl'
        if arch.startswith("stable-cascade"):
            return 'sc'
        if arch.startswith("flux"):
            return 'f1'
        if arch.startswith("hunyuan-video"):
            return 'hv'

        if "v1-5" in str(self.metadata.get('ss_sd_model_name', "")):
            return 'sd1'
        if str(self.metadata.get('ss_v2', "")) == "True":
            return 'sd2'
        if 'flux' in self.name.lower():
            return 'f1'
        if 'xl' in self.name.lower():
            return 'xl'

        return ''

    def set_hash(self, v):
        self.hash = v or ''
        self.shorthash = self.hash[0:8]

    def read_hash(self):
        if not self.hash:
            self.set_hash(hashes.sha256(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or '')

    def get_info(self):
        data = {}
        if shared.cmd_opts.no_metadata:
            return data
        if self.filename is not None:
            fn = os.path.splitext(self.filename)[0] + '.json'
            if os.path.exists(fn):
                data = shared.readfile(fn, silent=True)
                if type(data) is list:
                    data = data[0]
        return data

    def get_desc(self):
        if shared.cmd_opts.no_metadata:
            return None
        if self.filename is not None:
            fn = os.path.splitext(self.filename)[0] + '.txt'
            if os.path.exists(fn):
                return shared.readfile(fn, silent=True)
        return None

    def get_alias(self):
        if shared.opts.lora_preferred_name == "filename":
            return self.name
        else:
            return self.alias


class Network:  # LoraModule
    def __init__(self, name, network_on_disk: NetworkOnDisk):
        self.name = name
        self.network_on_disk = network_on_disk
        self.te_multiplier = 1.0
        self.unet_multiplier = [1.0] * 3
        self.dyn_dim = None
        self.modules = {}
        self.bundle_embeddings = {}
        self.mtime = None
        self.mentioned_name = None
        self.tags = None
        """the text that was used to add the network to prompt - can be either name or an alias"""


class ModuleType:
    def create_module(self, net: Network, weights: NetworkWeights) -> Union[Network, None]: # pylint: disable=W0613
        return None


class NetworkModule:
    def __init__(self, net: Network, weights: NetworkWeights):
        self.network = net
        self.network_key = weights.network_key
        self.sd_key = weights.sd_key
        self.sd_module = weights.sd_module
        if hasattr(self.sd_module, 'weight'):
            if hasattr(self.sd_module, "sdnq_dequantizer"):
                self.shape = self.sd_module.sdnq_dequantizer.original_shape
            else:
                self.shape = self.sd_module.weight.shape
        self.dim = None
        self.bias = weights.w.get("bias")
        self.alpha = weights.w["alpha"].item() if "alpha" in weights.w else None
        self.scale = weights.w["scale"].item() if "scale" in weights.w else None
        self.dora_scale = weights.w.get("dora_scale", None)
        self.dora_norm_dims = len(self.shape) - 1

    def multiplier(self):
        unet_multiplier = 3 * [self.network.unet_multiplier] if not isinstance(self.network.unet_multiplier, list) else self.network.unet_multiplier
        if 'transformer' in self.sd_key[:20]:
            return self.network.te_multiplier
        if "down_blocks" in self.sd_key:
            return unet_multiplier[0]
        if "mid_block" in self.sd_key:
            return unet_multiplier[1]
        if "up_blocks" in self.sd_key:
            return unet_multiplier[2]
        else:
            return unet_multiplier[0]

    def calc_scale(self):
        if self.scale is not None:
            return self.scale
        if self.dim is not None and self.alpha is not None:
            return self.alpha / self.dim
        return 1.0

    def apply_weight_decompose(self, updown, orig_weight):
        # Match the device/dtype
        orig_weight = orig_weight.to(updown.dtype)
        dora_scale = self.dora_scale.to(device=orig_weight.device, dtype=updown.dtype)
        updown = updown.to(orig_weight.device)

        merged_scale1 = updown + orig_weight
        merged_scale1_norm = (
            merged_scale1.transpose(0, 1)
            .reshape(merged_scale1.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(merged_scale1.shape[1], *[1] * self.dora_norm_dims)
            .transpose(0, 1)
        )

        dora_merged = (
                merged_scale1 * (dora_scale / merged_scale1_norm)
        )
        final_updown = dora_merged - orig_weight
        return final_updown

    def finalize_updown(self, updown, orig_weight, output_shape, ex_bias=None):
        if self.bias is not None:
            updown = updown.reshape(self.bias.shape)
            updown += self.bias.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = updown.reshape(output_shape)
        if len(output_shape) == 4:
            updown = updown.reshape(output_shape)
        if orig_weight.size().numel() == updown.size().numel():
            updown = updown.reshape(orig_weight.shape)
        if ex_bias is not None:
            ex_bias = ex_bias * self.multiplier()
        if self.dora_scale is not None:
            updown = self.apply_weight_decompose(updown, orig_weight)
        return updown * self.calc_scale() * self.multiplier(), ex_bias

    def calc_updown(self, target):
        raise NotImplementedError

    def forward(self, x, y):
        raise NotImplementedError
