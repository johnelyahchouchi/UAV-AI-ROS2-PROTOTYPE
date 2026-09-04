"""Versioned, bounded framing over mutually authenticated TLS 1.3."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import ssl
import struct
import threading
import time
from typing import Any, Iterable, Mapping, TypeAlias

from .config import DEFAULT_ALLOWED_CIDRS, PROTOCOL_VERSION, SecurityConfigurationError, SecurityLimits
from .detection import DetectionValidationError, parse_header_json, validate_header_envelope


ALPN_PROTOCOL = f"uav-frame/{PROTOCOL_VERSION}"
IPNetwork: TypeAlias = ipaddress.IPv4Network | ipaddress.IPv6Network


class ProtocolError(ValueError):
    """Raised when network input violates the versioned frame protocol."""


class ConnectionClosed(EOFError):
    """Raised when a peer disconnects before the requested bytes arrive."""


class ReplayError(ProtocolError):
    """Raised for duplicate, old, cross-session, or previously used sequences."""


@dataclass(frozen=True)
class TLSFiles:
    certificate: Path
    private_key: Path
    ca_certificate: Path


@dataclass(frozen=True)
class FramePacket:
    header: dict[str, Any]
    jpeg: bytes


def _required_file(env: Mapping[str, str], name: str) -> Path:
    raw = env.get(name, "").strip()
    if not raw:
        raise SecurityConfigurationError(f"{name} is required; insecure TCP is disabled")
    path = Path(raw).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SecurityConfigurationError(f"{name} does not identify a readable file") from error
    if not resolved.is_file():
        raise SecurityConfigurationError(f"{name} must identify a regular file")
    return resolved


def server_tls_files(env: Mapping[str, str] | None = None) -> TLSFiles:
    values = os.environ if env is None else env
    return TLSFiles(
        certificate=_required_file(values, "UAV_BRIDGE_TLS_CERT"),
        private_key=_required_file(values, "UAV_BRIDGE_TLS_KEY"),
        ca_certificate=_required_file(values, "UAV_BRIDGE_TLS_CA"),
    )


def client_tls_files(env: Mapping[str, str] | None = None) -> TLSFiles:
    values = os.environ if env is None else env
    return TLSFiles(
        certificate=_required_file(values, "UAV_SENDER_TLS_CERT"),
        private_key=_required_file(values, "UAV_SENDER_TLS_KEY"),
        ca_certificate=_required_file(values, "UAV_BRIDGE_TLS_CA"),
    )


def create_server_tls_context(files: TLSFiles) -> ssl.SSLContext:
    """Create a TLS 1.3-only server context that requires a trusted client cert."""

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_cert_chain(str(files.certificate), str(files.private_key))
    context.load_verify_locations(cafile=str(files.ca_certificate))
    context.set_alpn_protocols([ALPN_PROTOCOL])
    return context


def create_client_tls_context(files: TLSFiles) -> ssl.SSLContext:
    """Create a TLS 1.3-only client context with hostname and CA verification."""

    context = ssl.create_default_context(
        ssl.Purpose.SERVER_AUTH, cafile=str(files.ca_certificate)
    )
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_cert_chain(str(files.certificate), str(files.private_key))
    context.set_alpn_protocols([ALPN_PROTOCOL])
    return context


def validate_negotiated_tls(connection: ssl.SSLSocket) -> None:
    """Fail when TLS/ALPN/client authentication does not match protocol policy."""

    if connection.version() != "TLSv1.3":
        raise ProtocolError("TLS 1.3 is required")
    if connection.selected_alpn_protocol() != ALPN_PROTOCOL:
        raise ProtocolError("Peer did not negotiate the UAV frame protocol")
    if not connection.getpeercert():
        raise ProtocolError("Peer did not present a verified certificate")


def parse_allowed_cidrs(raw: str | None) -> tuple[IPNetwork, ...]:
    """Parse a comma-separated peer allowlist, requiring at least one network."""

    value = DEFAULT_ALLOWED_CIDRS if raw is None or not raw.strip() else raw
    networks = []
    for item in value.split(","):
        try:
            networks.append(ipaddress.ip_network(item.strip(), strict=False))
        except ValueError as error:
            raise SecurityConfigurationError(
                f"UAV_BRIDGE_ALLOWED_CIDRS contains an invalid network: {item!r}"
            ) from error
    if not networks:
        raise SecurityConfigurationError("UAV_BRIDGE_ALLOWED_CIDRS cannot be empty")
    return tuple(networks)


def peer_is_allowed(peer_ip: str, networks: Iterable[IPNetwork]) -> bool:
    try:
        address = ipaddress.ip_address(peer_ip)
    except ValueError:
        return False
    return any(address.version == network.version and address in network for network in networks)


def recv_exact(connection: Any, size: int, *, maximum: int) -> bytes:
    """Receive a validated bounded byte count using an efficient bytearray."""

    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ProtocolError("Receive size must be a positive integer")
    if size > maximum:
        raise ProtocolError(f"Receive size {size} exceeds the configured limit {maximum}")
    data = bytearray()
    while len(data) < size:
        try:
            chunk = connection.recv(size - len(data))
        except socket.timeout as error:
            raise ProtocolError("Timed out while receiving a frame") from error
        except ssl.SSLError as error:
            raise ProtocolError("TLS rejected unauthenticated or corrupted frame data") from error
        if not chunk:
            raise ConnectionClosed("Peer disconnected during a frame")
        data.extend(chunk)
    return bytes(data)


def encode_packet(
    header: Mapping[str, Any],
    jpeg: bytes,
    *,
    limits: SecurityLimits | None = None,
) -> bytes:
    """Serialize one bounded protocol packet; TLS provides transport AEAD."""

    budget = limits or SecurityLimits.from_environment()
    if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= budget.max_jpeg_size:
        raise ProtocolError("JPEG size is outside the configured limit")
    try:
        header_bytes = json.dumps(
            dict(header), allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProtocolError("Header is not finite JSON") from error
    if not 0 < len(header_bytes) <= budget.max_header_size:
        raise ProtocolError("Header size is outside the configured limit")
    return struct.pack("!I", len(header_bytes)) + header_bytes + jpeg


def receive_packet(
    connection: Any,
    *,
    limits: SecurityLimits | None = None,
) -> FramePacket:
    """Read one packet, rejecting attacker lengths before payload reads."""

    budget = limits or SecurityLimits.from_environment()
    prefix = recv_exact(connection, 4, maximum=4)
    header_size = struct.unpack("!I", prefix)[0]
    if not 0 < header_size <= budget.max_header_size:
        raise ProtocolError("Header length is outside the configured limit")
    header_bytes = recv_exact(connection, header_size, maximum=budget.max_header_size)
    try:
        header = parse_header_json(header_bytes)
        envelope = validate_header_envelope(header, limits=budget)
    except DetectionValidationError as error:
        raise ProtocolError(str(error)) from error
    jpeg = recv_exact(connection, envelope["jpeg_size"], maximum=budget.max_jpeg_size)
    return FramePacket(header=header, jpeg=jpeg)


class SessionReplayCache:
    """Bound memory while preventing reuse of prior application session IDs."""

    def __init__(self, maximum_sessions: int = 4096) -> None:
        if maximum_sessions <= 0:
            raise ValueError("maximum_sessions must be positive")
        self._maximum = maximum_sessions
        self._seen: set[str] = set()
        self._order: deque[str] = deque()
        self._lock = threading.Lock()

    def claim(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._seen:
                raise ReplayError("Session ID was already used by a prior TLS connection")
            self._seen.add(session_id)
            self._order.append(session_id)
            while len(self._order) > self._maximum:
                self._seen.remove(self._order.popleft())


class SessionSequenceValidator:
    """Enforce one session identity and strictly increasing sequences per TLS connection."""

    def __init__(self, replay_cache: SessionReplayCache) -> None:
        self._replay_cache = replay_cache
        self.session_id: str | None = None
        self.last_sequence = -1

    def check(self, header: Mapping[str, Any]) -> int:
        session_id = str(header["session_id"])
        sequence = int(header["seq"])
        if self.session_id is None:
            self._replay_cache.claim(session_id)
            self.session_id = session_id
        elif session_id != self.session_id:
            raise ReplayError("Session ID changed within one TLS connection")
        if sequence <= self.last_sequence:
            raise ReplayError("Frame sequence is duplicate or older than the last accepted frame")
        return sequence

    def commit(self, sequence: int) -> None:
        if sequence <= self.last_sequence:
            raise ReplayError("Cannot commit a duplicate or older frame sequence")
        self.last_sequence = sequence


def new_session_id() -> str:
    """Return 256 bits of fresh application-session material."""

    return secrets.token_hex(32)


def make_frame_header(
    *,
    session_id: str,
    sequence: int,
    width: int,
    height: int,
    jpeg_size: int,
    detections: list[dict[str, Any]],
    timestamp: float | None = None,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session_id,
        "seq": sequence,
        "timestamp": time.time() if timestamp is None else timestamp,
        "source_width": width,
        "source_height": height,
        "jpeg_size": jpeg_size,
        "detections": detections,
    }


def connect_tls_sender(
    host: str,
    port: int,
    *,
    context: ssl.SSLContext,
    server_name: str | None = None,
    timeout: float = 5.0,
) -> ssl.SSLSocket:
    """Connect and verify the bridge certificate before returning a usable socket."""

    raw = socket.create_connection((host, port), timeout=timeout)
    try:
        connection = context.wrap_socket(raw, server_hostname=server_name or host)
        connection.settimeout(timeout)
        validate_negotiated_tls(connection)
        return connection
    except Exception:
        raw.close()
        raise
