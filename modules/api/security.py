"""Security utilities for the API layer.

Centralizes path confinement, URL validation, WebSocket ticket auth,
and rate limiting so multiple endpoints can reuse the same logic.
"""

import os
import time
import secrets
import socket
import ipaddress
import threading
import collections
from pathlib import Path
from urllib.parse import urlparse
from fastapi.exceptions import HTTPException
from modules.logger import log


# ---------------------------------------------------------------------------
# Path confinement
# ---------------------------------------------------------------------------

def is_confined_to(filepath: str, allowed_roots: list[str]) -> bool:
    """Check whether *filepath* resolves inside one of *allowed_roots*."""
    try:
        resolved = Path(filepath).resolve()
        for root in allowed_roots:
            root_resolved = Path(root).resolve()
            if resolved == root_resolved or resolved.is_relative_to(root_resolved):
                return True
    except (OSError, ValueError):
        pass
    return False


# ---------------------------------------------------------------------------
# SSRF-safe URL validation
# ---------------------------------------------------------------------------

ALLOWED_DOWNLOAD_DOMAINS = {
    'civitai.com',
    'huggingface.co',
    'github.com',
    'githubusercontent.com',
    'raw.githubusercontent.com',
}

PRIVATE_NETWORKS = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('169.254.0.0/16'),
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('fc00::/7'),
    ipaddress.ip_network('fe80::/10'),
]


def _is_private_ip(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
        return any(addr in net for net in PRIVATE_NETWORKS)
    except ValueError:
        return True  # unparseable → treat as private


def _domain_matches_allowed(hostname: str) -> bool:
    hostname = hostname.lower().rstrip('.')
    for domain in ALLOWED_DOWNLOAD_DOMAINS:
        if hostname == domain or hostname.endswith('.' + domain):
            return True
    return False


def validate_download_url(url: str):
    """Raise HTTPException(400) if *url* is not a safe download target.

    Domain allowlist and private-IP blocking are only enforced when
    ``--listen`` is set (server is network-exposed).  Local users can
    download from any source without restrictions.
    """
    from modules import shared
    parsed = urlparse(url)
    # Scheme check
    if parsed.scheme == 'http':
        if parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise HTTPException(status_code=400, detail="Only HTTPS URLs are allowed for remote downloads")
    elif parsed.scheme != 'https':
        raise HTTPException(status_code=400, detail=f"Unsupported URL scheme: {parsed.scheme}")
    hostname = (parsed.hostname or '').lower()
    if not hostname:
        raise HTTPException(status_code=400, detail="Invalid URL: no hostname")
    # Domain allowlist + private IP blocking only when network-exposed
    if shared.cmd_opts.listen:
        if not _domain_matches_allowed(hostname):
            raise HTTPException(status_code=400, detail=f"Downloads from '{hostname}' are not allowed")
        try:
            for info in socket.getaddrinfo(hostname, parsed.port or 443, proto=socket.IPPROTO_TCP):
                ip_str = info[4][0]
                if _is_private_ip(ip_str):
                    raise HTTPException(status_code=400, detail="URL resolves to a private/internal address")
        except socket.gaierror:
            raise HTTPException(status_code=400, detail=f"Cannot resolve hostname: {hostname}") from None


# ---------------------------------------------------------------------------
# WebSocket ticket store (single-use, short-lived tokens)
# ---------------------------------------------------------------------------

class TicketStore:
    def __init__(self, ttl: int = 60):
        self.ttl = ttl
        self._tickets: dict[str, float] = {}  # ticket -> expiry timestamp
        self._lock = threading.Lock()

    def create(self) -> str:
        ticket = secrets.token_urlsafe(32)
        with self._lock:
            self._tickets[ticket] = time.time() + self.ttl
            self._cleanup()
        return ticket

    def validate(self, ticket: str) -> bool:
        with self._lock:
            expiry = self._tickets.pop(ticket, None)
        if expiry is None:
            return False
        return time.time() < expiry

    def _cleanup(self):
        now = time.time()
        expired = [t for t, exp in self._tickets.items() if now >= exp]
        for t in expired:
            self._tickets.pop(t, None)


ws_tickets = TicketStore(ttl=60)


# ---------------------------------------------------------------------------
# Rate limiter (per-IP sliding window)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._windows: dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            dq = self._windows.get(ip)
            if dq is None:
                dq = collections.deque()
                self._windows[ip] = dq
            # Purge timestamps outside the window
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.max_requests:
                return False
            dq.append(now)
            return True


generation_limiter = RateLimiter(max_requests=10, window_seconds=60)


# ---------------------------------------------------------------------------
# /js asset roots (computed once, cached)
# ---------------------------------------------------------------------------

_js_allowed_roots: list[str] | None = None


def get_js_allowed_roots() -> list[str]:
    """Return the list of directories from which /js may serve files."""
    global _js_allowed_roots
    if _js_allowed_roots is not None:
        return _js_allowed_roots
    from modules import paths
    roots = [
        paths.script_path,                      # project root (javascript/, html/, frontend/dist/)
        os.path.join(paths.script_path, 'extensions-builtin'),
        paths.extensions_dir,                    # user extensions
    ]
    _js_allowed_roots = [str(Path(r).resolve()) for r in roots if r]
    log.debug(f'Security: /js allowed roots={_js_allowed_roots}')
    return _js_allowed_roots
