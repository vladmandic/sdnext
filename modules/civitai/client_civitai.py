import os
import time
from modules.logger import log
from modules.civitai.basemodels_civitai import fetch_github_base_models
from modules.civitai.models_civitai import CivitModel, CivitVersion, CivitImage, CivitSearchResponse, CivitTagResponse, CivitCreatorResponse, CivitUserProfile


options_cache: dict = {}
options_cache_time: float = 0
OPTIONS_TTL = 3600  # 1 hour


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

    def _get(self, path: str, params: dict | None = None, token: str | None = None, stream: bool = False):
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
        return shared.req(url, headers=headers if headers else None, stream=stream)

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
            log.error(f'CivitAI search: code={r.status_code} reason={getattr(r, "reason", "")}')
            return CivitSearchResponse()
        data = r.json()
        if 'items' not in data:
            # single model by numeric query — wrap in search response
            try:
                model = CivitModel.parse_obj(data)
                return CivitSearchResponse(items=[model])
            except Exception:
                return CivitSearchResponse()
        try:
            return CivitSearchResponse.parse_obj(data)
        except Exception as e:
            log.error(f'CivitAI search parse error: {e}')
            return CivitSearchResponse()

    def get_model(self, model_id: int, *, token: str | None = None) -> CivitModel | None:
        r = self._get(f'/models/{model_id}', token=token)
        if r.status_code != 200:
            log.error(f'CivitAI get model: id={model_id} code={r.status_code}')
            return None
        try:
            return CivitModel.parse_obj(r.json())
        except Exception as e:
            log.error(f'CivitAI get model parse error: id={model_id} {e}')
            return None

    def get_version(self, version_id: int, *, token: str | None = None) -> CivitVersion | None:
        r = self._get(f'/model-versions/{version_id}', token=token)
        if r.status_code != 200:
            log.error(f'CivitAI get version: id={version_id} code={r.status_code}')
            return None
        try:
            return CivitVersion.parse_obj(r.json())
        except Exception as e:
            log.error(f'CivitAI get version parse error: id={version_id} {e}')
            return None

    def get_version_by_hash(self, hash_str: str, *, token: str | None = None) -> CivitVersion | None:
        r = self._get(f'/model-versions/by-hash/{hash_str}', token=token)
        if r.status_code != 200:
            return None
        try:
            return CivitVersion.parse_obj(r.json())
        except Exception as e:
            log.error(f'CivitAI get version by hash parse error: hash={hash_str} {e}')
            return None

    def get_images(self, *, model_version_id: int | None = None, limit: int | None = None, token: str | None = None) -> list[CivitImage]:
        params: dict = {}
        if model_version_id is not None:
            params['modelVersionId'] = model_version_id
        if limit is not None:
            params['limit'] = limit
        r = self._get('/images', params=params, token=token)
        if r.status_code != 200:
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
            return CivitCreatorResponse()
        try:
            return CivitCreatorResponse.parse_obj(r.json())
        except Exception as e:
            log.error(f'CivitAI get creators parse error: {e}')
            return CivitCreatorResponse()

    def get_me(self, token: str | None = None) -> CivitUserProfile | None:
        r = self._get('/me', token=token)
        if r.status_code != 200:
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

    def discover_options(self) -> dict:
        global options_cache, options_cache_time # pylint: disable=global-statement
        now = time.time()
        if options_cache and (now - options_cache_time) < OPTIONS_TTL:
            return options_cache
        from modules import shared
        result: dict = {'types': [], 'sort': [], 'period': [], 'base_models': [], 'base_models_info': []}
        # Send invalid params to trigger 400 with valid enum values in error response
        probes = [
            ('types', '/models', {'types': '__invalid__'}),
            ('sort', '/models', {'sort': '__invalid__'}),
            ('period', '/models', {'period': '__invalid__'}),
            ('base_models', '/images', {'baseModels': '__invalid__'}),  # /models no longer validates baseModels; /images still does
        ]
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
                    import json as _json
                    issues = error.get('issues', [])
                    if not issues:
                        try:
                            issues = _json.loads(error.get('message', '[]'))
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
        # Merge live probe names with github metadata. The probe is the source
        # of truth for which names exist; github provides per-entry metadata and
        # doubles as a fallback name list if the probe returned empty.
        github_entries = fetch_github_base_models()
        github_index: dict = {entry['name']: entry for entry in github_entries}
        probe_names: list = result['base_models']
        if not probe_names and github_entries:
            probe_names = [entry['name'] for entry in github_entries]
            result['base_models'] = probe_names
        result['base_models_info'] = [
            github_index.get(name, {'name': name, 'type': 'image', 'group': '', 'hidden': False})
            for name in probe_names
        ]
        options_cache = result
        options_cache_time = now
        log.debug(f'CivitAI options: types={len(result["types"])} sort={len(result["sort"])} period={len(result["period"])} base_models={len(result["base_models"])} (enriched={len(github_index)})')
        return result


client = CivitaiClient()
