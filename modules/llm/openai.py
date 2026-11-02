"""OpenAI-compatible inference with explicit remote modality discovery.

Recent llama-server/llama-swap model metadata is required. Discovery never probes
models with inference requests, and this adapter performs no lifecycle control.
"""

import base64
import io
import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request as HTTPRequest, urlopen
from modules.llm.backend import Backend, ModelRef, Request, Response


PREFIX = 'external/'
DEFAULT_MAX_TOKENS = 8192


def generation_parameters(max_tokens, temperature, sample=True, top_k=0, top_p=0, penalty=None, seed=None, min_tokens=0, num_beams=1, custom_args='', process_words='', default_max_tokens=None):
    """Translate supported caller options without silently dropping local features."""
    if min_tokens:
        raise ValueError('External LLM does not support the local minimum-token constraint')
    if num_beams != 1:
        raise ValueError('External LLM does not support local beam search')
    if custom_args or process_words:
        raise ValueError('External LLM does not support local custom arguments or logits processing')
    if max_tokens is None:
        max_tokens = max(DEFAULT_MAX_TOKENS, default_max_tokens or 0)
    if type(max_tokens) not in (int, float) or max_tokens < 1:
        raise ValueError('External LLM output token limit must be positive')
    result = {'max_tokens': int(max_tokens), 'temperature': float(temperature) if sample else 0.0}
    if top_k > 0:
        result['top_k'] = int(top_k)
    if top_p > 0:
        result['top_p'] = float(top_p)
    if penalty is not None:
        result['repeat_penalty'] = float(penalty)
    if seed is not None:
        result['seed'] = int(seed)
    return result


@dataclass(frozen=True)
class ModelInfo:
    ref: ModelRef
    name: str
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    context_length: int | None = None
    # No standardized request-side reasoning capability is currently advertised.
    thinking: bool | None = None

    @property
    def selection(self):
        return PREFIX + quote(self.ref.model_id, safe='')

    @property
    def vision(self):
        return 'image' in self.input_modalities

    @property
    def text_generation(self):
        return 'text' in self.input_modalities and 'text' in self.output_modalities


class OpenAIBackend(Backend):
    def __init__(self, base_url: str, api_key: str = '', timeout: float = 120):
        url = urlsplit(base_url.strip())
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('External LLM URL must be an HTTP(S) API base URL without credentials, query, or fragment')
        if timeout <= 0:
            raise ValueError('External LLM timeout must be positive')
        self.base_url = base_url.strip().rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.models: tuple[ModelInfo, ...] = ()
        self.discovered = False

    def _request(self, path: str, payload: dict | None = None) -> dict:
        headers = {'Accept': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        if data is not None:
            headers['Content-Type'] = 'application/json'
        request = HTTPRequest(f'{self.base_url}/{path}', data=data, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as e:
            # Do not echo request bodies, credentials, or untrusted server error text.
            raise RuntimeError(f'External LLM {path}: HTTP {e.code}') from None
        except (URLError, TimeoutError, OSError) as e:
            raise RuntimeError(f'External LLM {path}: connection failed ({type(e).__name__})') from None
        except (ValueError, UnicodeError):
            raise ValueError(f'External LLM {path}: invalid JSON response') from None
        if not isinstance(result, dict):
            raise ValueError(f'External LLM {path}: expected a JSON object')
        return result

    def list_models(self) -> tuple[ModelInfo, ...]:
        # Failed refreshes must not leave stale capability information selectable.
        self.models, self.discovered = (), False
        result = self._request('models')
        if not isinstance(result.get('data'), list):
            raise ValueError('External LLM discovery: missing model list')
        models = []
        seen = set()
        for entry in result['data']:
            model_id = entry.get('id') if isinstance(entry, dict) else None
            if not isinstance(model_id, str) or not model_id or model_id in seen:
                raise ValueError('External LLM discovery: model IDs must be nonempty and unique')
            architecture = entry.get('architecture')
            if not isinstance(architecture, dict):
                raise ValueError(f'External LLM discovery: model "{model_id}" is missing architecture modality metadata')
            modalities = []
            for key in ('input_modalities', 'output_modalities'):
                values = architecture.get(key)
                if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
                    raise ValueError(f'External LLM discovery: model "{model_id}" is missing {key}')
                modalities.append(tuple(values))
            context_length = entry.get('context_length')
            if context_length is not None and (type(context_length) is not int or context_length <= 0):
                raise ValueError(f'External LLM discovery: model "{model_id}" has an invalid context length')
            name = entry.get('name')
            models.append(ModelInfo(ModelRef('openai', model_id), name if isinstance(name, str) and name else model_id, input_modalities=modalities[0], output_modalities=modalities[1], context_length=context_length))
            seen.add(model_id)
        self.models, self.discovered = tuple(models), True
        return self.models

    def model_info(self, model: ModelRef) -> ModelInfo:
        if model.provider != 'openai':
            raise ValueError(f'External backend cannot run provider "{model.provider}"')
        if not self.discovered:
            self.list_models()
        info = next((info for info in self.models if info.ref == model), None)
        if info is None:
            raise ValueError(f'External LLM model "{model.model_id}" is not advertised by the configured endpoint')
        return info

    def generate(self, model: ModelRef, request: Request) -> Response:
        info = self.model_info(model)
        if not info.text_generation:
            raise ValueError('Selected external model does not advertise text generation')
        if request.thinking:
            raise ValueError('External model does not advertise request-side thinking controls')
        if request.image is not None and not info.vision:
            raise ValueError('Selected external model does not advertise image input')
        parameters = dict(request.parameters)
        parameters.setdefault('max_tokens', DEFAULT_MAX_TOKENS)
        allowed = {'max_tokens', 'temperature', 'top_p', 'top_k', 'repeat_penalty', 'seed', 'stop'}
        unsupported = parameters.keys() - allowed
        if unsupported:
            raise ValueError(f'Unsupported external LLM parameters: {", ".join(sorted(unsupported))}')
        if info.context_length and parameters.get('max_tokens', 0) >= info.context_length:
            raise ValueError('External LLM output token limit must be smaller than its advertised serving context')
        content = request.prompt
        if request.image is not None:
            with io.BytesIO() as stream:
                request.image.convert('RGB').save(stream, format='JPEG')
                encoded = base64.b64encode(stream.getvalue()).decode('ascii')
            content = [
                {'type': 'text', 'text': request.prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encoded}'}},
            ]
        messages = []
        if request.system:
            messages.append({'role': 'system', 'content': request.system})
        messages.append({'role': 'user', 'content': content})
        if request.prefill:
            messages.append({'role': 'assistant', 'content': request.prefill})
        result = self._request('chat/completions', {'model': model.model_id, 'messages': messages, 'stream': False, **parameters})
        choices = result.get('choices')
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict) or not isinstance(choices[0].get('message'), dict):
            raise ValueError('External LLM response is missing a completion message')
        message = choices[0]['message']
        text = message.get('content') or ''
        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        if not isinstance(text, str) or not isinstance(reasoning, str):
            raise ValueError('External LLM response content and reasoning must be text')
        if not text.strip():
            reason = choices[0].get('finish_reason')
            raise ValueError(f'External LLM returned no answer (finish reason: {reason}, output budget: {parameters["max_tokens"]}); increase the token budget if reasoning used it')
        usage = result.get('usage') or {}
        if not isinstance(usage, dict):
            raise ValueError('External LLM response usage must be an object')
        return Response(text=text.strip(), reasoning=reasoning.strip(), usage=usage)
