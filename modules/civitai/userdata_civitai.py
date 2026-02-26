import os
import json
import threading
from datetime import datetime
from modules.logger import log


def _data_dir() -> str:
    from modules import paths
    return paths.data_path or paths.script_path


class UserList:
    def __init__(self, filename: str):
        self._filename = filename
        self._items: list[str] = []
        self._lock = threading.Lock()
        self._load()

    def _path(self) -> str:
        return os.path.join(_data_dir(), self._filename)

    def _load(self):
        path = self._path()
        if os.path.isfile(path):
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._items = data
            except Exception as e:
                log.warning(f'CivitAI userdata load error: file={path} {e}')

    def _save(self):
        path = self._path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._items, f, indent=2)
        except Exception as e:
            log.error(f'CivitAI userdata save error: file={path} {e}')

    def add(self, name: str) -> bool:
        with self._lock:
            if name not in self._items:
                self._items.append(name)
                self._save()
                return True
            return False

    def remove(self, name: str) -> bool:
        with self._lock:
            if name in self._items:
                self._items.remove(name)
                self._save()
                return True
            return False

    def list(self) -> list[str]:
        with self._lock:
            return list(self._items)

    def contains(self, name: str) -> bool:
        with self._lock:
            return name in self._items


class SearchHistory:
    def __init__(self, filename: str, max_entries: int = 30):
        self._filename = filename
        self._max_entries = max_entries
        self._entries: list[dict] = []
        self._lock = threading.Lock()
        self._load()

    def _path(self) -> str:
        return os.path.join(_data_dir(), self._filename)

    def _load(self):
        path = self._path()
        if os.path.isfile(path):
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._entries = data
            except Exception as e:
                log.warning(f'CivitAI search history load error: file={path} {e}')

    def _save(self):
        path = self._path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._entries, f, indent=2)
        except Exception as e:
            log.error(f'CivitAI search history save error: file={path} {e}')

    def add(self, search_type: str, term: str):
        with self._lock:
            entry = {
                "type": search_type,
                "term": term,
                "timestamp": datetime.now().isoformat(),
            }
            # Remove duplicate if same type+term exists
            self._entries = [e for e in self._entries if not (e.get('type') == search_type and e.get('term') == term)]
            self._entries.insert(0, entry)
            # Trim to max
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[:self._max_entries]
            self._save()

    def list(self, search_type: str = None) -> list[dict]:
        with self._lock:
            if search_type:
                return [e for e in self._entries if e.get('type') == search_type]
            return list(self._entries)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._save()


bookmarks = UserList("civitai_bookmarks.json")
banned = UserList("civitai_banned.json")
search_history = SearchHistory("civitai_search_history.json")
