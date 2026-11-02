"""Adapters around SD.Next's existing in-process and cloud inference paths.

SDK imports remain lazy: importing this module must not initialize shared state,
install dependencies, authenticate, load a model, or make network requests.
"""

from collections.abc import Callable
from urllib.parse import unquote
from modules.llm.backend import Backend, ModelRef, Request, Response


def resolve_model(model_id: str) -> ModelRef:
    """Translate legacy catalog IDs at the boundary, before backend dispatch.

    External selections carry an explicit prefix so remote model names cannot
    select a local or cloud handler.
    """
    if model_id.startswith('external/'):
        return ModelRef('openai', unquote(model_id.removeprefix('external/')))
    if model_id.startswith(('google/gemini-', 'gemini-')):
        return ModelRef('gemini', model_id.removeprefix('google/'))
    if model_id.startswith(('xai/grok-', 'grok-')):
        return ModelRef('grok', model_id.removeprefix('xai/'))
    return ModelRef('inprocess', model_id)


class InProcessBackend(Backend):
    """Keep task/model-specific tokenization and loading in existing handlers."""

    def __init__(self, generate: Callable[[str, Request], Response]):
        self._generate = generate

    def generate(self, model: ModelRef, request: Request) -> Response:
        if model.provider != 'inprocess':
            raise ValueError(f'In-process backend cannot run provider "{model.provider}"')
        return self._generate(model.model_id, request)


class GeminiBackend(Backend):
    def generate(self, model: ModelRef, request: Request) -> Response:
        if model.provider != 'gemini':
            raise ValueError(f'Gemini backend cannot run provider "{model.provider}"')
        from modules.caption import gemini
        text = gemini.predict(request.prompt, request.image, model.model_id, request.system, request.prefill, request.thinking, request.parameters)
        return Response(text=text)


class GrokBackend(Backend):
    def generate(self, model: ModelRef, request: Request) -> Response:
        if model.provider != 'grok':
            raise ValueError(f'Grok backend cannot run provider "{model.provider}"')
        from modules.caption import grok
        text = grok.predict(request.prompt, request.image, model.model_id, request.system, request.prefill, request.thinking, request.parameters)
        return Response(text=text)


def get_backend(model: ModelRef, local_generate: Callable[[str, Request], Response] | None = None) -> Backend:
    if model.provider == 'inprocess':
        if local_generate is None:
            raise ValueError('In-process backend requires a local inference handler')
        return InProcessBackend(local_generate)
    if model.provider == 'gemini':
        return GeminiBackend()
    if model.provider == 'grok':
        return GrokBackend()
    if model.provider == 'openai':
        from modules.llm.catalog import catalog
        return catalog.backend(discover=True)
    raise ValueError(f'Unknown LLM provider "{model.provider}"')
