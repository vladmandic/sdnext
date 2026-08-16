from __future__ import annotations

from typing import Union
import os
from collections import UserDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from modules.logger import log

do_cache_folders = os.environ.get('SD_NO_CACHE', None) is None

FilePathList = list[str]
FilePathIterator = Iterator[str]
DirectoryPathList = list[str]
DirectoryPathIterator = Iterator[str]
DirectoryList = list['Directory']
DirectoryIterator = Iterator['Directory']
DirectoryCollection = dict[str, 'Directory']
ExtensionFilter = Callable
ExtensionList = list[str]
RecursiveType = Union[bool, Callable]


@lru_cache(maxsize=1024)
def real_path(directory_path: str) -> str | None:
    """Cached real_path resolution to avoid repeated abspath/expanduser calls."""
    if not directory_path:
        return None
    try:
        return os.path.abspath(os.path.expanduser(directory_path))
    except Exception:
        return None


@dataclass(frozen=True)
class Directory:
    path: str = field(default_factory=str)
    files: FilePathList = field(default_factory=list)
    directories: DirectoryPathList = field(default_factory=list)
    mtime: float = 0.0

    @classmethod
    def from_dict(cls, dict_object: dict) -> Directory:
        directory = cls.__new__(cls)
        object.__setattr__(directory, 'path', dict_object.get('path'))
        object.__setattr__(directory, 'mtime', dict_object.get('mtime', 0.0))
        object.__setattr__(directory, 'files', dict_object.get('files', []))
        object.__setattr__(directory, 'directories', dict_object.get('directories', []))
        return directory

    def clear(self) -> None:
        self._update(Directory.from_dict({
            'path': None,
            'mtime': 0.0,
            'files': [],
            'directories': []
        }))

    def update(self, source_directory: Directory) -> Directory:
        if source_directory is not self and source_directory is not None:
            self._update(source_directory)
        return self

    def _update(self, source: Directory) -> None:
        assert not source.path or source.path == self.path, (
            f'When updating a directory, the paths must match. '
            f'Attempted to update Directory `{self.path}` with `{source.path}`'
        )
        for dead_path in self.directories:
            if dead_path not in source.directories:
                delete_cached_directory(dead_path)
        self.directories[:] = source.directories
        self.files[:] = source.files
        object.__setattr__(self, 'mtime', source.mtime)

    @property
    def exists(self) -> bool:
        return bool(self.path and os.path.exists(self.path))

    @property
    def is_directory(self) -> bool:
        return bool(self.path and os.path.isdir(self.path))

    @property
    def live_mtime(self) -> float:
        try:
            return os.path.getmtime(self.path) if self.path else 0.0
        except OSError:
            return 0.0

    @property
    def is_stale(self) -> bool:
        return self.mtime != self.live_mtime


class DirectoryCache(UserDict, DirectoryCollection):
    def __delattr__(self, directory_path: str) -> None:
        directory: Directory = get_directory(directory_path, fetch=False)
        if directory:
            for child in directory.directories:
                delete_cached_directory(child)
            directory.clear()
        self.data.pop(directory_path, None)


def clean_directory(directory: Directory, /, recursive: RecursiveType = False) -> bool:
    if not directory.is_directory:
        delete_cached_directory(directory.path)
        return False

    is_clean = not directory.is_stale
    if not is_clean:
        fetched = fetch_directory(directory.path)
        if fetched:
            directory.update(fetched)
    elif recursive:
        for directory_path in list(directory.directories):
            try:
                recurse = recursive and (not callable(recursive) or recursive(directory.path))
                child_dir = get_directory(directory_path, fetch=recurse)
                if child_dir:
                    if child_dir.is_directory:
                        if recurse:
                            is_clean = clean_directory(child_dir, recursive=recurse) and is_clean
                        continue
                    delete_cached_directory(directory_path)
                if recurse:
                    directory.directories.remove(directory_path)
                is_clean = False
            except Exception:
                pass
    return is_clean


def get_directory(directory_or_path: str | Directory, /, fetch: bool = True) -> Directory | None:
    if isinstance(directory_or_path, Directory):
        if directory_or_path.is_directory:
            return directory_or_path
        directory_or_path = directory_or_path.path

    resolved_path = real_path(directory_or_path)
    if not resolved_path:
        return None

    if resolved_path not in cache_folders:
        if fetch:
            directory = fetch_directory(directory_path=resolved_path)
            if directory and do_cache_folders:
                cache_folders[resolved_path] = directory
            return directory
        return None

    cached = cache_folders[resolved_path]
    clean_directory(cached)
    return cache_folders.get(resolved_path)


def fetch_directory(directory_path: str) -> Directory | None:
    for directory in _walk(directory_path, recurse=False):
        return directory
    return None


def _walk(top: str, recurse: RecursiveType = True) -> Iterator[Directory]:
    nondirs = []
    walk_dirs = []
    top_mtime = 0.0

    try:
        top_mtime = os.path.getmtime(top)
        scandir_it = os.scandir(top)
    except OSError:
        return

    with scandir_it:
        for entry in scandir_it:
            if not entry.is_dir(follow_symlinks=True):
                nondirs.append(entry.path)
            else:
                if entry.is_symlink() and not os.path.exists(entry.path):
                    log.error(f'Files broken symlink: {entry.path}')
                else:
                    walk_dirs.append(entry.path)

    yield Directory(path=top, files=nondirs, directories=walk_dirs, mtime=top_mtime)

    if recurse:
        for new_path in walk_dirs:
            if callable(recurse) and not recurse(new_path):
                continue
            yield from _walk(new_path, recurse=recurse)


def _cached_walk(top: str, recurse: RecursiveType = True) -> Iterator[Directory]:
    top_dir = get_directory(top)
    if not top_dir:
        return
    yield top_dir
    if recurse:
        for child_directory in top_dir.directories:
            if os.path.basename(child_directory).startswith('models--'):
                continue
            if callable(recurse) and not recurse(child_directory):
                continue
            yield from _cached_walk(child_directory, recurse=recurse)


def walk(top: str, recurse: RecursiveType = True, cached: bool = True) -> Iterator[Directory]:
    yield from _cached_walk(top, recurse=recurse) if cached else _walk(top, recurse=recurse)


def delete_cached_directory(directory_path: str) -> bool:
    if directory_path in cache_folders:
        del cache_folders[directory_path]
        return True
    return False


def is_directory(dir_path: str) -> bool:
    return bool(dir_path and os.path.isdir(dir_path))


def directory_mtime(directory_path: str, /, recursive: RecursiveType = True) -> float:
    dirs = get_directories(directory_path, recursive=recursive)
    return max((d.mtime for d in dirs), default=0.0)


def unique_directories(directories: DirectoryPathList, /, recursive: RecursiveType = True) -> DirectoryPathIterator:
    directories = sorted(unique_paths(directories), reverse=True)
    while directories:
        directory = directories.pop()
        yield directory
        if not recursive:
            continue
        _directory = os.path.join(directory, '')
        child_directory = None
        while directories and directories[-1].startswith(_directory):
            if not callable(recursive) or not child_directory:
                directories.pop()
                continue
            child_directory = directories[-1][len(directory):]
            if child_directory:
                next_directory = _directory
                _remove_directory = None
                for sub_directory in child_directory.split(os.path.sep):
                    next_directory = os.path.join(next_directory, sub_directory)
                    if recursive(next_directory):
                        _remove_directory = os.path.join(next_directory, '')
                        break
                while _remove_directory and directories:
                    if not directories[-1].startswith(_remove_directory):
                        break
                    directories.pop()


def unique_paths(directory_paths: DirectoryPathList) -> DirectoryPathIterator:
    seen = set()
    for path in directory_paths:
        if path:
            r = real_path(path)
            if r and r not in seen:
                seen.add(r)
                yield r


def get_directories(*directory_paths: DirectoryPathList, fetch: bool = True, recursive: RecursiveType = True) -> list[Directory]:
    dirs = unique_directories(directory_paths, recursive=recursive)
    return [d for d in (get_directory(p, fetch=fetch) for p in dirs) if d]


def directory_files(*directories_or_paths: DirectoryPathList | DirectoryList, recursive: RecursiveType = True) -> FilePathIterator:
    """Iterative directory file gatherer avoiding deeply nested generator recursion."""
    visited = set()
    stack = list(directories_or_paths)

    while stack:
        item = stack.pop()
        dir_obj = get_directory(item) if not isinstance(item, Directory) else item
        if not dir_obj or dir_obj.path in visited:
            continue

        visited.add(dir_obj.path)
        yield from dir_obj.files

        if recursive:
            for child_path in dir_obj.directories:
                if callable(recursive) and not recursive(child_path):
                    continue
                stack.append(child_path)


def extension_filter(ext_filter: ExtensionList | None = None, ext_blacklist: ExtensionList | None = None) -> ExtensionFilter:
    """Fast C-level tuple.endswith checks."""
    valid_exts = tuple(ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in ext_filter) if ext_filter else None
    black_exts = tuple(ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in ext_blacklist) if ext_blacklist else None

    def filter_function(fp: str) -> bool:
        fp_lower = fp.lower()
        if valid_exts and not fp_lower.endswith(valid_exts):
            return False
        if black_exts and fp_lower.endswith(black_exts):
            return False
        return True

    return filter_function


def not_hidden(filepath: str) -> bool:
    return not os.path.basename(filepath).startswith('.')


def filter_files(file_paths: FilePathList, ext_filter: ExtensionList | None = None, ext_blacklist: ExtensionList | None = None) -> FilePathIterator:
    return filter(extension_filter(ext_filter, ext_blacklist), file_paths)


def list_files(*directory_paths: DirectoryPathList, ext_filter: ExtensionList | None = None, ext_blacklist: ExtensionList | None = None, recursive: RecursiveType = True) -> FilePathIterator:
    raw_files = directory_files(*directory_paths, recursive=recursive)
    return filter_files(raw_files, ext_filter, ext_blacklist)


cache_folders = DirectoryCache({})
