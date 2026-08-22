"""Built-in backends, registered in ascending priority."""
from modules.attention.registry import registry
from modules.attention.backends import dynamic, flex, triton_amd, flash_ck, sage, sdnq

registry.register(dynamic.backend)
registry.register(flex.backend)
registry.register(triton_amd.backend)
registry.register(flash_ck.backend)
registry.register(sage.backend)
registry.register(sdnq.backend)
