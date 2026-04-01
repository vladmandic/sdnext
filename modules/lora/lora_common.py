import os
from modules.lora import lora_timers
from modules.lora import network_lora, network_hada, network_ia3, network_oft, network_lokr, network_full, network_norm, network_glora


timer = lora_timers.Timer()
debug = os.environ.get('SD_LORA_DEBUG', None) is not None
module_types = [
    network_lora.ModuleTypeLora(),
    network_hada.ModuleTypeHada(),
    network_ia3.ModuleTypeIa3(),
    network_oft.ModuleTypeOFT(),
    network_lokr.ModuleTypeLokr(),
    network_full.ModuleTypeFull(),
    network_norm.ModuleTypeNorm(),
    network_glora.ModuleTypeGLora(),
]
loaded_networks: list = [] # no type due to circular import
previously_loaded_networks: list = [] # no type due to circular import
extra_network_lora = None # initialized in extra_networks.py
