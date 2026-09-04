from __future__ import annotations

import json
import socket
import ssl
import struct

import pytest

from uav_security.config import SecurityLimits
from uav_security.transport import (
    ConnectionClosed,
    ProtocolError,
    ReplayError,
    SessionReplayCache,
    SessionSequenceValidator,
    encode_packet,
    parse_allowed_cidrs,
    peer_is_allowed,
    receive_packet,
    recv_exact,
)


class MemorySocket:
    def __init__(self, data: bytes, chunk_size: int | None = None):
        self.data = bytearray(data)
        self.chunk_size = chunk_size
        self.recv_calls = 0

    def recv(self, size: int) -> bytes:
        self.recv_calls += 1
        if not self.data:
            return b""
        count = min(size, len(self.data), self.chunk_size or size)
        result = bytes(self.data[:count])
        del self.data[:count]
        return result


def packet_bytes(header: dict[str, object], jpeg: bytes = b"jpeg") -> bytes:
    encoded = json.dumps(header).encode("utf-8")
    return struct.pack("!I", len(encoded)) + encoded + jpeg


def test_valid_bounded_frame_round_trip(valid_header):
    wire = encode_packet(valid_header, b"jpeg")
    packet = receive_packet(MemorySocket(wire, chunk_size=3))
    assert packet.header == valid_header
    assert packet.jpeg == b"jpeg"


def test_ffffffff_header_is_rejected_before_payload_read():
    connection = MemorySocket(struct.pack("!I", 0xFFFFFFFF))
    with pytest.raises(ProtocolError, match="Header length"):
        receive_packet(connection)
    assert connection.recv_calls == 1


def test_zero_length_header_is_rejected():
    with pytest.raises(ProtocolError, match="Header length"):
        receive_packet(MemorySocket(struct.pack("!I", 0)))


def test_negative_jpeg_size_is_rejected_before_payload_read(valid_header):
    valid_header["jpeg_size"] = -1
    connection = MemorySocket(packet_bytes(valid_header, b"secret-payload"))
    with pytest.raises(ProtocolError, match="jpeg_size"):
        receive_packet(connection)
    assert bytes(connection.data) == b"secret-payload"


def test_oversized_jpeg_is_rejected_before_payload_read(valid_header):
    valid_header["jpeg_size"] = 5
    limits = SecurityLimits(max_jpeg_size=4)
    connection = MemorySocket(packet_bytes(valid_header, b"12345"))
    with pytest.raises(ProtocolError, match="jpeg_size"):
        receive_packet(connection, limits=limits)
    assert bytes(connection.data) == b"12345"


def test_truncated_frame_is_rejected(valid_header):
    valid_header["jpeg_size"] = 10
    with pytest.raises(ConnectionClosed):
        receive_packet(MemorySocket(packet_bytes(valid_header, b"short")))


def test_malformed_json_is_rejected():
    malformed = b"{not-json"
    wire = struct.pack("!I", len(malformed)) + malformed
    with pytest.raises(ProtocolError, match="JSON"):
        receive_packet(MemorySocket(wire))


def test_recv_exact_rejects_invalid_sizes():
    for size in (0, -1, True, 5):
        with pytest.raises(ProtocolError):
            recv_exact(MemorySocket(b"abcde"), size, maximum=4)


def test_tls_authentication_error_is_fail_closed():
    class CorruptTLSSocket:
        def recv(self, _size):
            raise ssl.SSLError("bad record mac")

    with pytest.raises(ProtocolError, match="TLS rejected"):
        recv_exact(CorruptTLSSocket(), 4, maximum=4)


def test_peer_outside_allowlist_is_rejected():
    networks = parse_allowed_cidrs("10.42.0.0/24,127.0.0.1/32")
    assert peer_is_allowed("10.42.0.9", networks)
    assert not peer_is_allowed("10.43.0.9", networks)
    assert not peer_is_allowed("not-an-ip", networks)


def test_sequences_must_strictly_increase(valid_header):
    tracker = SessionSequenceValidator(SessionReplayCache())
    assert tracker.check(valid_header) == 1
    tracker.commit(1)
    valid_header["seq"] = 2
    assert tracker.check(valid_header) == 2
    tracker.commit(2)
    with pytest.raises(ReplayError, match="duplicate or older"):
        tracker.check(valid_header)
    valid_header["seq"] = 1
    with pytest.raises(ReplayError, match="duplicate or older"):
        tracker.check(valid_header)


def test_prior_session_replay_is_rejected(valid_header):
    cache = SessionReplayCache()
    first = SessionSequenceValidator(cache)
    first.commit(first.check(valid_header))
    replay = SessionSequenceValidator(cache)
    with pytest.raises(ReplayError, match="prior TLS connection"):
        replay.check(valid_header)


def test_session_id_cannot_change_mid_connection(valid_header):
    tracker = SessionSequenceValidator(SessionReplayCache())
    tracker.commit(tracker.check(valid_header))
    valid_header["seq"] = 2
    valid_header["session_id"] = "b" * 64
    with pytest.raises(ReplayError, match="changed"):
        tracker.check(valid_header)


def test_idle_peer_times_out_and_accept_loop_can_continue():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    first_client = socket.create_connection(listener.getsockname(), timeout=1)
    first_server, _ = listener.accept()
    first_server.settimeout(0.05)
    with pytest.raises(ProtocolError, match="Timed out"):
        recv_exact(first_server, 4, maximum=4)
    first_server.close()
    first_client.close()

    second_client = socket.create_connection(listener.getsockname(), timeout=1)
    second_server, _ = listener.accept()
    second_client.sendall(b"next")
    assert recv_exact(second_server, 4, maximum=4) == b"next"
    second_server.close()
    second_client.close()
    listener.close()
