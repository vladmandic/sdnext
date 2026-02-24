"""Secure secrets storage for API tokens and credentials.

Secrets are stored in a dedicated file (secrets.json) separate from
config.json and are never exposed to the frontend or settings API.
Environment variables take highest priority.
"""

import os
import stat
from modules.json_helpers import readfile, writefile
from modules.logger import log


_secrets: dict[str, str] = {}
_secrets_path: str = ''
_initialized: bool = False


def init(config_filename: str):
    """Initialize secrets manager. Derives secrets.json path from config path."""
    global _secrets, _secrets_path, _initialized # pylint: disable=global-statement
    config_dir = os.path.dirname(os.path.abspath(config_filename))
    _secrets_path = os.path.join(config_dir, 'secrets.json')
    if os.path.isfile(_secrets_path):
        _secrets = readfile(_secrets_path, silent=True, lock=True, as_type="dict")
        log.debug(f'Secrets: loaded={len(_secrets)} file="{_secrets_path}"')
    else:
        _secrets = {}
        log.debug(f'Secrets: file="{_secrets_path}" not found, starting empty')
    _initialized = True


def _persist():
    """Write secrets to disk and restrict file permissions."""
    if not _secrets_path:
        return
    writefile(_secrets, _secrets_path, silent=True)
    try:
        if os.name != 'nt':
            os.chmod(_secrets_path, stat.S_IRUSR | stat.S_IWUSR)  # 600
    except OSError as err:
        log.warning(f'Secrets: cannot set file permissions: {err}')


def get(key: str, env_var: str | None = None) -> str:
    """Get a secret value. Priority: env var > secrets.json > empty string."""
    if env_var:
        val = os.environ.get(env_var, '')
        if val:
            return val
    return _secrets.get(key, '')


def set(key: str, value: str): # pylint: disable=redefined-builtin
    """Store a secret and persist to disk."""
    if not value:
        _secrets.pop(key, None)
    else:
        _secrets[key] = value
    _persist()
    log.debug(f'Secrets: set key="{key}" configured={bool(value)}')


def has(key: str, env_var: str | None = None) -> bool:
    """Check if a secret is configured from any source."""
    return bool(get(key, env_var))


def get_mask(key: str, env_var: str | None = None) -> str:
    """Return a masked representation of the secret for display."""
    val = get(key, env_var)
    if not val:
        return ''
    if len(val) <= 8:
        return '*' * len(val)
    return val[:3] + '...' + val[-4:]


def get_status(key: str, env_var: str | None = None) -> dict:
    """Return status dict for a secret: configured, source, masked."""
    if env_var and os.environ.get(env_var, ''):
        return {'configured': True, 'source': 'env', 'masked': get_mask(key, env_var)}
    if _secrets.get(key, ''):
        return {'configured': True, 'source': 'file', 'masked': get_mask(key, env_var)}
    return {'configured': False, 'source': 'none', 'masked': ''}


class SecretStr(str):
    """A str subclass that redacts itself in repr() but passes the real value to APIs.

    When a dict containing a SecretStr is printed (e.g. in f-string log
    messages like ``f"config={some_dict}"``), Python calls ``repr()`` on
    each value, so the token appears as ``'***'`` in logs.  Direct string
    operations (``f"Bearer {token}"``, concatenation, header building)
    use the real value via the normal ``str`` interface.
    """

    def __repr__(self):
        return "'***'"


_REDACT_KEYS = frozenset({'token', 'api_key', 'api_token', 'access_token', 'secret', 'password', 'authorization'})
_REDACTED = '***'


def sanitize_dict(d: dict) -> dict:
    """Return a shallow copy of *d* with secret-looking values replaced by '***'."""
    return {k: (_REDACTED if k in _REDACT_KEYS and v else v) for k, v in d.items()}


def migrate_from_config(config_data: dict, secret_keys: dict[str, str | None]) -> list[str]:
    """One-time migration of secrets from config.json data.

    Args:
        config_data: The loaded config.json dict.
        secret_keys: Mapping of secret key -> env_var name.

    Returns:
        List of keys that were migrated (and should be removed from config_data).
    """
    migrated = []
    for key, _env_var in secret_keys.items():
        val = config_data.get(key, '')
        if val and isinstance(val, str) and val.strip():
            if not _secrets.get(key, ''):
                _secrets[key] = val.strip()
                log.info(f'Secrets: migrated "{key}" from config.json')
            migrated.append(key)
    if migrated:
        _persist()
    return migrated
