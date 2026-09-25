"""Refuse outbound network use from this process.

Outbound TCP connects and name lookups are refused. The server may bind
loopback, or 0.0.0.0 / :: when you explicitly open the page on the LAN.
llama.cpp is a child process; it is started with ``--offline`` so this
guard is not the only control.
"""

from __future__ import annotations

import ipaddress
import socket

class AirgapError(RuntimeError):
    pass


_INSTALLED = False
_ORIGINAL_CONNECT = None
_ORIGINAL_CONNECT_EX = None
_ORIGINAL_GETADDRINFO = None


def loopback_host(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    text = str(host).strip().strip("[]").lower()
    if text in {"localhost", "::1"}:
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def unspecified_host(host) -> bool:
    """0.0.0.0 and :: mean 'listen on every interface', not a remote peer."""
    if host is None:
        return False
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    text = str(host).strip().strip("[]").lower()
    if text in {"0.0.0.0", "::"}:
        return True
    try:
        return ipaddress.ip_address(text).is_unspecified
    except ValueError:
        return False


def _allowed_address(address) -> bool:
    # AF_UNIX addresses are a path string, not a network peer.
    if not isinstance(address, tuple) or not address:
        return True
    return loopback_host(address[0])


def install_airgap() -> None:
    global _INSTALLED, _ORIGINAL_CONNECT, _ORIGINAL_CONNECT_EX, _ORIGINAL_GETADDRINFO
    if _INSTALLED:
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        if not _allowed_address(address):
            host = address[0] if isinstance(address, tuple) and address else address
            raise AirgapError(f"blocked outbound connection to {host}")
        return original_connect(self, address)

    def connect_ex(self, address):
        if not _allowed_address(address):
            host = address[0] if isinstance(address, tuple) and address else address
            raise AirgapError(f"blocked outbound connection to {host}")
        return original_connect_ex(self, address)

    def getaddrinfo(host, port, *args, **kwargs):
        if not loopback_host(host) and not unspecified_host(host):
            raise AirgapError(f"blocked name lookup for {host}")
        return original_getaddrinfo(host, port, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo
    _ORIGINAL_CONNECT = original_connect
    _ORIGINAL_CONNECT_EX = original_connect_ex
    _ORIGINAL_GETADDRINFO = original_getaddrinfo
    _INSTALLED = True


def uninstall_airgap() -> None:
    global _INSTALLED, _ORIGINAL_CONNECT, _ORIGINAL_CONNECT_EX, _ORIGINAL_GETADDRINFO
    if not _INSTALLED:
        return
    socket.socket.connect = _ORIGINAL_CONNECT
    socket.socket.connect_ex = _ORIGINAL_CONNECT_EX
    socket.getaddrinfo = _ORIGINAL_GETADDRINFO
    _ORIGINAL_CONNECT = None
    _ORIGINAL_CONNECT_EX = None
    _ORIGINAL_GETADDRINFO = None
    _INSTALLED = False
