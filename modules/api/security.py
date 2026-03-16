"""Security utilities for the API layer.

Centralizes path confinement, URL validation, and WebSocket ticket auth
so multiple endpoints can reuse the same logic.
"""

import secrets
import socket
import time
import ipaddress
import threading
from pathlib import Path
from urllib.parse import urlparse
from fastapi.exceptions import HTTPException


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
        return True  # unparseable -> treat as private


def validate_download_url(url: str):
    """Raise HTTPException(400) if *url* is not a safe download target.

    HTTPS is required for remote URLs.  Private-IP blocking is only
    enforced when ``--listen`` is set (server is network-exposed).
    Local users can download from any non-private source without
    domain restrictions.
    """
    from modules import shared
    parsed = urlparse(url)
    if parsed.scheme == 'http':
        if parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise HTTPException(status_code=400, detail="Only HTTPS URLs are allowed for remote downloads")
    elif parsed.scheme != 'https':
        raise HTTPException(status_code=400, detail=f"Unsupported URL scheme: {parsed.scheme}")
    hostname = (parsed.hostname or '').lower()
    if not hostname:
        raise HTTPException(status_code=400, detail="Invalid URL: no hostname")
    if shared.cmd_opts.listen:
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
