import os
import json
import time
import threading
from types import SimpleNamespace
from modules.logger import log
from modules.civitai.models_civitai import CivitModel, CivitVersion, CivitVersionMini, CivitImage, CivitSearchResponse, CivitTagResponse, CivitCreatorResponse, CivitUserProfile


options_cache: dict = {}
options_cache_time: float = 0
OPTIONS_TTL = 3600  # 1 hour
# Civitai nsfwLevel bitmask: 1=PG/None 2=PG-13/Soft 4=R/Mature 8=X 16=XXX 32=Blocked
NSFW_LEVEL_SFW = 3   # None + Soft: Civitai's SFW browsing boundary
NSFW_LEVEL_ALL = 63  # every level set: disables filtering
BY_HASH_IDS_LIMIT = 10000  # POST /model-versions/by-hash/ids request cap
BY_HASH_LIMIT = 100  # POST /model-versions/by-hash request cap
MODEL_IDS_LIMIT = 100  # GET /models page cap; longer ids lists paginate
RETRY_LIMIT = 4  # retries after HTTP 429
RETRY_DELAY_MAX = 60  # seconds
request_slots: threading.BoundedSemaphore | None = None
request_slots_lock = threading.Lock()


def get_request_slots() -> threading.BoundedSemaphore:
    """Process-wide cap on concurrent CivitAI API requests, sized to shared.max_workers."""
    global request_slots # pylint: disable=global-statement
    with request_slots_lock:
        if request_slots is None:
            from modules.shared import max_workers
            request_slots = threading.BoundedSemaphore(max_workers)
    return request_slots


def retry_delay(response, attempt: int) -> float:
    """Seconds before retrying a 429: Retry-After when given in seconds, otherwise exponential."""
    headers = getattr(response, 'headers', None) or {}
    try:
        delay = float(headers.get('Retry-After'))
    except (TypeError, ValueError):
        delay = 2 ** attempt
    return min(max(delay, 0.0), RETRY_DELAY_MAX)


def response_message(response) -> str:
    """CivitAI error text from a failed response: its error string, ZodError issues, or the HTTP reason."""
    try:
        body = response.json()
    except Exception:
        body = None
    error = body.get('error') if isinstance(body, dict) else None
    if isinstance(error, dict):
        error = error.get('message', '')
        try:
            error = '; '.join(f"{'.'.join(str(p) for p in issue.get('path', []))}: {issue.get('message', '')}" for issue in json.loads(error))
        except Exception:
            pass
    message = body.get('message') if isinstance(body, dict) else None
    if isinstance(error, str) and isinstance(message, str) and message and message != error: # download refusals carry a short error and a longer message
        error = f'{error}: {message}'
    if not error:
        error = getattr(response, 'reason', '') or getattr(response, 'text', '')
    return str(error).strip()[:200]


def post_json(url: str, body, headers: dict):
    """POST with the timeout, TLS and failure shape of shared.req."""
    import requests
    try:
        return requests.post(url, json=body, timeout=30, headers=headers, verify=False, allow_redirects=True)
    except Exception as e:
        log.error(f'HTTP request error: url={url} {e}')
        return SimpleNamespace(status_code=500, text=f'HTTP request error: url={url} {e}')


class CivitaiClient:
    BASE_URL = "https://civitai.com/api/v1"

    def _get_token(self, token: str | None = None) -> str | None:
        if token:
            return token
        from modules import shared
        tok = getattr(shared.opts, 'civitai_token', '') or ''
        if tok:
            return tok
        return os.environ.get('CIVITAI_TOKEN', None)

    def send(self, method: str, path: str, params: dict | None = None, body=None, token: str | None = None, stream: bool = False):
        from modules import shared
        url = f"{self.BASE_URL}{path}"
        headers = {}
        tok = self._get_token(token)
        if tok:
            headers['Authorization'] = f'Bearer {tok}'
        if params:
            from urllib.parse import urlencode
            query = urlencode({k: v for k, v in params.items() if v is not None and v != ''}, doseq=True)
            if query:
                url = f"{url}?{query}"
        attempt = 0
        while True:
            with get_request_slots():
                if method == 'POST':
                    r = post_json(url, body, headers)
                else:
                    r = shared.req(url, headers=headers if headers else None, stream=stream)
            retry_after = (getattr(r, 'headers', None) or {}).get('Retry-After')
            if not (r.status_code == 429 or (r.status_code == 503 and retry_after is not None)) or attempt >= RETRY_LIMIT: # CivitAI sends 503 with Retry-After when search is overloaded
                return r
            delay = retry_delay(r, attempt)
            log.warning(f'CivitAI retry: path={path} code={r.status_code} attempt={attempt + 1} delay={delay:.0f}s message="{response_message(r)}"')
            time.sleep(delay)
            attempt += 1

    def _get(self, path: str, params: dict | None = None, token: str | None = None, stream: bool = False):
        return self.send('GET', path, params=params, token=token, stream=stream)

    def search_models(self, *, query: str = "", tag: str = "", types: str = "", sort: str = "", period: str = "",
                      base_models: list[str] | None = None, nsfw: bool | None = None, limit: int = 20,
                      cursor: str | None = None, username: str = "", favorites: bool = False,
                      token: str | None = None) -> CivitSearchResponse:
        params: dict = {}
        if query:
            params['query'] = query
        if tag:
            params['tag'] = tag
        if types:
            params['types'] = types
        if sort:
            params['sort'] = sort
        if period:
            params['period'] = period
        if base_models:
            params['baseModels'] = base_models
        if nsfw is not None:
            params['nsfw'] = 'true' if nsfw else 'false'
        if limit:
            params['limit'] = limit
        if cursor:
            params['cursor'] = cursor
        if username:
            params['username'] = username
        if favorites:
            params['favorites'] = 'true'
        r = self._get('/models', params=params, token=token)
        if r.status_code != 200:
            message = response_message(r)
            log.error(f'CivitAI search: code={r.status_code} message="{message}"')
            return CivitSearchResponse(error=message)
        data = r.json()
        if 'items' not in data:
            # single model by numeric query — wrap in search response
            try:
                model = CivitModel.parse_obj(data)
                return CivitSearchResponse(items=[model])
            except Exception:
                return CivitSearchResponse()
        try:
            response = CivitSearchResponse.parse_obj(data)
        except Exception as e:
            log.error(f'CivitAI search parse error: {e}')
            return CivitSearchResponse(error='search response could not be parsed')
        # /models rejects server-side level filtering and its nsfw boolean leaks
        # Mature+ content, so filter on each model's aggregate nsfwLevel here:
        # nsfw on keeps every level, nsfw off/unset keeps SFW (None + Soft).
        level = NSFW_LEVEL_ALL if nsfw else NSFW_LEVEL_SFW
        if level < NSFW_LEVEL_ALL:
            response.items = [m for m in response.items if (m.nsfw_level & ~level) == 0]
        return response

    def get_model(self, model_id: int, *, token: str | None = None) -> CivitModel | None:
        r = self._get(f'/models/{model_id}', token=token)
        if r.status_code != 200:
            log.error(f'CivitAI get model: id={model_id} code={r.status_code} message="{response_message(r)}"')
            return None
        try:
            return CivitModel.parse_obj(r.json())
        except Exception as e:
            log.error(f'CivitAI get model parse error: id={model_id} {e}')
            return None

    def get_version(self, version_id: int, *, token: str | None = None) -> CivitVersion | None:
        r = self._get(f'/model-versions/{version_id}', token=token)
        if r.status_code != 200:
            log.error(f'CivitAI get version: id={version_id} code={r.status_code} message="{response_message(r)}"')
            return None
        try:
            return CivitVersion.parse_obj(r.json())
        except Exception as e:
            log.error(f'CivitAI get version parse error: id={version_id} {e}')
            return None

    def get_version_by_hash(self, hash_str: str, *, token: str | None = None) -> CivitVersion | None:
        r = self._get(f'/model-versions/by-hash/{hash_str}', token=token)
        if r.status_code != 200:
            if r.status_code != 404:
                log.error(f'CivitAI get version by hash: hash={hash_str} code={r.status_code} message="{response_message(r)}"')
            return None
        try:
            return CivitVersion.parse_obj(r.json())
        except Exception as e:
            log.error(f'CivitAI get version by hash parse error: hash={hash_str} {e}')
            return None

    def get_version_mini(self, version_id: int, *, token: str | None = None) -> CivitVersionMini | None:
        r = self._get(f'/model-versions/mini/{version_id}', token=token)
        if r.status_code != 200:
            log.error(f'CivitAI get version mini: id={version_id} code={r.status_code} message="{response_message(r)}"')
            return None
        try:
            return CivitVersionMini.parse_obj(r.json())
        except Exception as e:
            log.error(f'CivitAI get version mini parse error: id={version_id} {e}')
            return None

    def get_version_ids_by_hash(self, hashes: list[str], *, token: str | None = None) -> tuple[list[dict], dict[str, int]]:
        """{modelVersionId, modelId, hash} rows for SHA256 hashes, plus the status code for each hash whose request failed."""
        rows, failed = [], {}
        for i in range(0, len(hashes), BY_HASH_IDS_LIMIT):
            chunk = hashes[i:i + BY_HASH_IDS_LIMIT]
            r = self.send('POST', '/model-versions/by-hash/ids', body=chunk, token=token)
            if r.status_code != 200:
                log.error(f'CivitAI version ids by hash: count={len(chunk)} code={r.status_code} message="{response_message(r)}"')
                failed.update(dict.fromkeys(chunk, r.status_code))
                continue
            try:
                rows.extend(r.json())
            except Exception as e:
                log.error(f'CivitAI version ids by hash parse error: count={len(chunk)} {e}')
                failed.update(dict.fromkeys(chunk, 500))
        return rows, failed

    def get_versions_by_hash(self, hashes: list[str], *, token: str | None = None) -> tuple[list[CivitVersion], dict[str, int]]:
        """Full versions for SHA256 hashes, plus the status code for each hash whose request failed."""
        versions, failed = [], {}
        for i in range(0, len(hashes), BY_HASH_LIMIT):
            chunk = hashes[i:i + BY_HASH_LIMIT]
            r = self.send('POST', '/model-versions/by-hash', body=chunk, token=token)
            if r.status_code != 200:
                log.error(f'CivitAI versions by hash: count={len(chunk)} code={r.status_code} message="{response_message(r)}"')
                failed.update(dict.fromkeys(chunk, r.status_code))
                continue
            try:
                versions.extend([CivitVersion.parse_obj(v) for v in r.json()])
            except Exception as e:
                log.error(f'CivitAI versions by hash parse error: count={len(chunk)} {e}')
                failed.update(dict.fromkeys(chunk, 500))
        return versions, failed

    def get_models_raw(self, model_ids: list[int], *, token: str | None = None) -> tuple[dict[int, dict], dict[int, int]]:
        """Unparsed /models items keyed by id, plus the status code for each id whose request failed."""
        models, failed = {}, {}
        for i in range(0, len(model_ids), MODEL_IDS_LIMIT):
            chunk = model_ids[i:i + MODEL_IDS_LIMIT]
            params = {'ids': ','.join(str(m) for m in chunk), 'limit': MODEL_IDS_LIMIT, 'nsfw': 'true'} # ids query drops NSFW models unless nsfw=true
            r = self.send('GET', '/models', params=params, token=token)
            if r.status_code != 200:
                log.error(f'CivitAI models by id: count={len(chunk)} code={r.status_code} message="{response_message(r)}"')
                failed.update(dict.fromkeys(chunk, r.status_code))
                continue
            try:
                for item in r.json().get('items', []):
                    models[item['id']] = item
            except Exception as e:
                log.error(f'CivitAI models by id parse error: count={len(chunk)} {e}')
                failed.update(dict.fromkeys(chunk, 500))
        return models, failed

    def get_images(self, *, model_version_id: int | None = None, limit: int | None = None, token: str | None = None) -> list[CivitImage]:
        params: dict = {}
        if model_version_id is not None:
            params['modelVersionId'] = model_version_id
        if limit is not None:
            params['limit'] = limit
        r = self._get('/images', params=params, token=token)
        if r.status_code != 200:
            log.error(f'CivitAI get images: code={r.status_code} message="{response_message(r)}"')
            return []
        data = r.json()
        items = data.get('items', [])
        result = []
        for item in items:
            try:
                result.append(CivitImage.parse_obj(item))
            except Exception:
                pass
        return result

    def get_images_raw(self, *, model_version_id: int | None = None, model_id: int | None = None, limit: int | None = None, token: str | None = None) -> list[dict]:
        params: dict = {}
        if model_version_id is not None:
            params['modelVersionId'] = model_version_id
        if model_id is not None:
            params['modelId'] = model_id
        if limit is not None:
            params['limit'] = limit
        r = self._get('/images', params=params, token=token)
        if r.status_code != 200:
            log.error(f'CivitAI get images: code={r.status_code} message="{response_message(r)}"')
            return []
        data = r.json()
        return data.get('items', [])

    def get_tags(self, *, query: str = "", limit: int = 20, page: int = 1) -> CivitTagResponse:
        params: dict = {}
        if query:
            params['query'] = query
        if limit:
            params['limit'] = limit
        if page > 1:
            params['page'] = page
        r = self._get('/tags', params=params)
        if r.status_code != 200:
            log.error(f'CivitAI get tags: code={r.status_code} message="{response_message(r)}"')
            return CivitTagResponse()
        try:
            return CivitTagResponse.parse_obj(r.json())
        except Exception as e:
            log.error(f'CivitAI get tags parse error: {e}')
            return CivitTagResponse()

    def get_creators(self, *, query: str = "", limit: int = 20, page: int = 1) -> CivitCreatorResponse:
        params: dict = {}
        if query:
            params['query'] = query
        if limit:
            params['limit'] = limit
        if page > 1:
            params['page'] = page
        r = self._get('/creators', params=params)
        if r.status_code != 200:
            log.error(f'CivitAI get creators: code={r.status_code} message="{response_message(r)}"')
            return CivitCreatorResponse()
        try:
            return CivitCreatorResponse.parse_obj(r.json())
        except Exception as e:
            log.error(f'CivitAI get creators parse error: {e}')
            return CivitCreatorResponse()

    def get_me(self, token: str | None = None) -> CivitUserProfile | None:
        r = self._get('/me', token=token)
        if r.status_code != 200:
            if r.status_code != 401:
                log.error(f'CivitAI get me: code={r.status_code} message="{response_message(r)}"')
            return None
        try:
            return CivitUserProfile.parse_obj(r.json())
        except Exception as e:
            log.error(f'CivitAI get me parse error: {e}')
            return None

    def validate_token(self, token: str) -> dict | None:
        """Validate a token by calling /me. Returns user info dict or None if invalid."""
        profile = self.get_me(token=token)
        if profile is None:
            return None
        return {"username": profile.username, "id": profile.id}

    def get_enums(self) -> dict:
        """Civitai enum lists (ModelType, ModelFileType, BaseModel, ActiveBaseModel, BaseModelType). Public endpoint."""
        r = self._get('/enums')
        if r.status_code != 200:
            log.debug(f'CivitAI enums: code={r.status_code} message="{response_message(r)}"')
            return {}
        try:
            return r.json()
        except Exception as e:
            log.debug(f'CivitAI enums parse error: {e}')
            return {}

    def discover_options(self) -> dict:
        global options_cache, options_cache_time # pylint: disable=global-statement
        now = time.time()
        if options_cache and (now - options_cache_time) < OPTIONS_TTL:
            return options_cache
        from modules import shared
        result: dict = {'types': [], 'sort': [], 'period': [], 'base_models': [], 'base_models_info': []}
        # Authoritative model-type and base-model lists from the documented /enums
        # endpoint. /enums carries no sort/period, so those are probed below.
        enums = self.get_enums()
        result['types'] = enums.get('ModelType', []) or []
        result['base_models'] = enums.get('BaseModel', []) or []
        # sort/period: /models still validates these, so recover the valid values
        # from its 400 (ZodError) response. types/base_models are only probed when
        # /enums was unavailable.
        probes = [
            ('sort', '/models', {'sort': '__invalid__'}),
            ('period', '/models', {'period': '__invalid__'}),
        ]
        if not result['types']:
            probes.append(('types', '/models', {'types': '__invalid__'}))
        if not result['base_models']:
            probes.append(('base_models', '/images', {'baseModels': '__invalid__'}))  # /models no longer validates baseModels; /images still does
        for key, path, params in probes:
            try:
                url = f"{self.BASE_URL}{path}"
                from urllib.parse import urlencode
                query = urlencode(params)
                full_url = f"{url}?{query}"
                r = shared.req(full_url)
                if r.status_code == 400:
                    data = r.json()
                    error = data.get('error', {})
                    if not isinstance(error, dict):
                        continue
                    # Parse ZodError: error.message is a JSON-encoded array of issues
                    issues = error.get('issues', [])
                    if not issues:
                        try:
                            issues = json.loads(error.get('message', '[]'))
                        except Exception:
                            issues = []
                    for issue in issues:
                        # Flat format: options directly on issue
                        options = issue.get('options', [])
                        if options:
                            result[key] = options
                            break
                        # Flat format: values directly on issue (sort/period use this)
                        values = issue.get('values', [])
                        if values:
                            result[key] = values
                            break
                        # Nested union format: errors[][].values
                        for err_group in issue.get('errors', []):
                            if isinstance(err_group, list):
                                for err in err_group:
                                    vals = err.get('values', [])
                                    if vals:
                                        result[key] = vals
                                        break
                            if result[key]:
                                break
                        if result[key]:
                            break
            except Exception as e:
                log.debug(f'CivitAI discover options: key={key} {e}')
        # hidden marks names in BaseModel but not in ActiveBaseModel, the retired set; an empty ActiveBaseModel hides nothing
        active = set(enums.get('ActiveBaseModel', []) or [])
        result['base_models_info'] = [
            {'name': name, 'type': 'image', 'group': '', 'hidden': bool(active) and name not in active}
            for name in result['base_models']
        ]
        options_cache = result
        options_cache_time = now
        log.debug(f'CivitAI options: types={len(result["types"])} sort={len(result["sort"])} period={len(result["period"])} base_models={len(result["base_models"])} active={len(active)}')
        return result


client = CivitaiClient()
