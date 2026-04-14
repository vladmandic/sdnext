import hashlib
import os.path
from collections import defaultdict
from typing import Literal, TypeAlias, TypedDict
from rich import progress, errors
from modules.logger import console
from modules.logger import log
from modules.json_helpers import readfile, writefile
from modules.paths import data_path


class HashEntry(TypedDict):
    mtime: float
    sha256: str


class HashStore(dict[str, HashEntry]):
    def __init__(self, *args):
        super().__init__(*args)

    def add_hash(self, key: str, mtime: float = 0, sha256: str | None = None):
        self.__setitem__(key, {"mtime": mtime, "sha256": sha256 or ""})


default_hash_store = "hashes"
KnownHashStores: TypeAlias = Literal["hashes", "hashes-addnet"]  # For autocomplete in IDE

cache_filename = os.path.join(data_path, "data", "cache.json")
progress_ok = True
# defaultdict allows for easily using new stores without needing to define them ahead of time
_data: defaultdict[str, HashStore] = defaultdict(HashStore)


def load_cache():
    if os.path.isfile(cache_filename):
        for store, data in readfile(cache_filename, lock=True, as_type="dict").items():
            _data[store] = HashStore(data)


def save_cache():
    # Don't include empty hash stores
    filtered = filter(lambda item: len(item[1]) > 0, _data.items())
    writefile(dict(filtered), cache_filename)


def cache(store: KnownHashStores | str | None = None) -> HashStore:
    if store is None:
        return _data[default_hash_store]
    return _data[store]


def calculate_sha256(filename, quiet=False):
    global progress_ok # pylint: disable=global-statement
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024
    if not quiet:
        if progress_ok:
            try:
                with progress.open(filename, 'rb', description=f'[cyan]Calculating hash: [yellow]{filename}', auto_refresh=True, console=console) as f:
                    for chunk in iter(lambda: f.read(blksize), b""):
                        hash_sha256.update(chunk)
            except errors.LiveError:
                log.warning('Hash: attempting to use function in a thread')
                progress_ok = False
        if not progress_ok:
            with open(filename, 'rb') as f:
                for chunk in iter(lambda: f.read(blksize), b""):
                    hash_sha256.update(chunk)
    else:
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def sha256_from_cache(filename: str, title: str, *, store: KnownHashStores | str | None = None):
    hashes = cache(store)
    if title not in hashes:
        return None
    cached = hashes[title]
    ondisk_mtime = os.path.getmtime(filename) if os.path.isfile(filename) else 0
    if ondisk_mtime > cached["mtime"] or not cached["sha256"]:
        return None
    return cached["sha256"]


def sha256(filename: str, title: str, *, store: KnownHashStores | str | None = None):
    from modules import shared
    global progress_ok # pylint: disable=global-statement
    hashes = cache(store)
    sha256_value = sha256_from_cache(filename, title, store=store)
    if sha256_value is not None:
        return sha256_value
    if shared.cmd_opts.no_hashing:
        return None
    if not os.path.isfile(filename):
        return None
    jobid = shared.state.begin("Hash")
    if store == "hashes-addnet":
        if progress_ok:
            try:
                with progress.open(filename, 'rb', description=f'[cyan]Calculating hash: [yellow]{filename}', auto_refresh=True, console=console) as f:
                    sha256_value = addnet_hash_safetensors(f)
            except errors.LiveError:
                log.warning('Hash: attempting to use function in a thread')
                progress_ok = False
        if not progress_ok:
            with open(filename, 'rb') as f:
                sha256_value = addnet_hash_safetensors(f)
    else:
        sha256_value = calculate_sha256(filename)
    hashes.add_hash(title, os.path.getmtime(filename), sha256_value)
    shared.state.end(jobid)
    save_cache()
    return sha256_value


def addnet_hash_safetensors(b):
    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024
    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")
    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)
    return hash_sha256.hexdigest()
