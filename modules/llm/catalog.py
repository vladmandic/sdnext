"""One configured external endpoint; cached discovery shared by all consumers."""

from threading import RLock
from urllib.parse import urlsplit
from modules.llm.openai import OpenAIBackend, PREFIX


class Catalog:
    def __init__(self):
        self._key = None
        self._backend = None
        self._lock = RLock()
        self._labels = {}

    def backend(self, discover=False) -> OpenAIBackend:
        from modules import shared
        key = (shared.opts.llm_api_url.strip(), shared.opts.llm_api_key, float(shared.opts.llm_api_timeout))
        if not key[0]:
            raise ValueError('Configure an external LLM API URL in Settings > Model Options')
        with self._lock:
            if key != self._key:
                self._backend = OpenAIBackend(*key)
                self._key = key
            if discover and not self._backend.discovered:
                self._backend.list_models()
            return self._backend

    def models(self):
        from modules import shared
        if not shared.opts.llm_api_url.strip():
            return ()
        try:
            return self.backend().models
        except ValueError:
            return () # Invalid configuration cannot expose the previous endpoint's models.

    def refresh(self):
        from modules import shared
        if not shared.opts.llm_api_url.strip():
            return ()
        with self._lock:
            return self.backend().list_models()

    def display_name(self, selection):
        """A UI alias only; backend and saved/API identities stay external/ IDs."""
        selection = self.selection(selection)
        if not isinstance(selection, str) or not selection.startswith(PREFIX):
            return selection
        if not any(model.selection == selection for model in self.models()):
            return selection
        url = urlsplit(self.backend().base_url)
        host = url.hostname
        if ':' in host:
            host = f'[{host}]'
        if url.port and url.port != (443 if url.scheme == 'https' else 80):
            host = f'{host}:{url.port}'
        # Spaces distinguish display labels from valid local/cloud repository
        # IDs even for a single-label hostname such as "google".
        label = f'{host} / {selection.removeprefix(PREFIX)}'
        # Retain previously displayed aliases until restart. An endpoint refresh
        # must not turn a still-visible UI choice into a local HF model request.
        with self._lock:
            self._labels[label] = selection
        return label

    def selection(self, value):
        with self._lock:
            return self._labels.get(value, value) if isinstance(value, str) else value


catalog = Catalog()


def is_external(selection):
    selection = catalog.selection(selection)
    return isinstance(selection, str) and selection.startswith(PREFIX)


def model_info(selection):
    selection = catalog.selection(selection)
    return next((model for model in catalog.models() if model.selection == selection), None)


def display_name(selection):
    return catalog.display_name(selection)


def resolve_display_name(value):
    return catalog.selection(value)


def selections(vision=False):
    return {model.selection: model for model in catalog.models() if model.text_generation and (not vision or model.vision)}


def refresh_models(tolerate_errors=False):
    """Explicit discovery for refresh controls and API model listings."""
    try:
        catalog.refresh()
    except Exception as e:
        if not tolerate_errors:
            raise
        from modules.logger import log
        log.error(f'External LLM discovery: {e}')


def initialize():
    """Optional pre-UI discovery; unavailable servers must not break startup."""
    from modules.logger import log
    try:
        models = catalog.refresh()
        if models:
            log.info(f'External LLM discovery: models={len(models)}')
    except Exception as e:
        log.error(f'External LLM discovery: {e}')
