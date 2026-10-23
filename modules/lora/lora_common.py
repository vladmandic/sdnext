import os
from modules.lora import lora_timers


timer = lora_timers.Timer()
debug = os.environ.get('SD_LORA_DEBUG', None) is not None
loaded_networks: list = [] # no type due to circular import
previously_loaded_networks: list = [] # no type due to circular import
extra_network_lora = None # initialized in extra_networks.py
last_backup_size: int = 0 # bytes of weight backups the last activate pass held
last_mode: str = '' # how that pass left the weights: backup, fuse or factor
pass_token = None # identity of the running activate walk; per-pass caches key on it and stay cold outside one
