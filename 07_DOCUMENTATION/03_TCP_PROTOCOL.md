# 03 - Authenticated TCP Frame Protocol

## Compatibility

The active sender and bridge use `PROTOCOL_VERSION = 2`. Version 2 is an
intentional breaking change: an old plaintext sender cannot connect to the new
bridge, and a new sender cannot use the old bridge. Deploy the sender,
`uav_security` package, and bridge together.

There is no automatic or opt-in plaintext fallback.

## Trust and TLS

The Windows sender is the TCP client and the ROS 2 bridge is the server. All
framing is carried inside TLS 1.3. Both endpoints present certificates issued by
the configured deployment CA:

- the sender verifies the bridge certificate and its DNS/IP subject alternative
  name;
- the bridge requires and verifies the sender client certificate;
- TLS record AEAD provides confidentiality, integrity, and protection against
  replaying ciphertext from a different TLS session;
- application session IDs and sequence numbers add explicit frame replay checks.

Certificates and private keys are external deployment files. They must not be
placed in this repository or printed in logs.

## Required configuration

| Variable | Used by | Required/default | Purpose |
|---|---|---|---|
| `UAV_BRIDGE_TLS_CERT` | bridge | required | Bridge certificate chain |
| `UAV_BRIDGE_TLS_KEY` | bridge | required | Bridge private key |
| `UAV_BRIDGE_TLS_CA` | both | required | CA certificate used to verify the peer |
| `UAV_SENDER_TLS_CERT` | sender | required | Sender client certificate chain |
| `UAV_SENDER_TLS_KEY` | sender | required | Sender client private key |
| `UAV_BRIDGE_TLS_SERVER_NAME` | sender | target IP | Expected bridge certificate DNS/IP identity |
| `UAV_BRIDGE_HOST` | sender | `127.0.0.1` | Bridge IP when `--target` is omitted |
| `UAV_BRIDGE_BIND_ADDRESS` | bridge | `127.0.0.1` | Local IP on which to listen |
| `UAV_BRIDGE_PORT` | both | `5010` | TCP port |
| `UAV_BRIDGE_ALLOWED_CIDRS` | bridge | `127.0.0.0/8,::1/128` | Comma-separated peer allowlist |

A cross-machine deployment must explicitly set the bind address and allowlist.
Use a host firewall as an additional control. A wildcard bind still requires
mutual TLS and does not bypass the allowlist.

## Wire format

After the TLS/ALPN handshake (`uav-frame/2`), each frame is:

```text
4-byte unsigned big-endian JSON length
bounded UTF-8 JSON header
exactly jpeg_size bytes of JPEG data
```

The required JSON fields are:

```json
{
  "protocol_version": 2,
  "session_id": "64 lowercase hexadecimal characters",
  "seq": 1,
  "timestamp": 1700000000.0,
  "source_width": 960,
  "source_height": 540,
  "jpeg_size": 123456,
  "detections": []
}
```

The sender creates 256 bits of random session material after each authenticated
connection. Within that connection, `session_id` cannot change and `seq` must be
strictly greater than the last accepted value. A recently used session ID cannot
be claimed by a new connection. Tracking resets only after a new authenticated,
previously unseen session is established.

## Validation and limits

Defaults are centralized in `uav_security/config.py`:

| Limit | Default |
|---|---:|
| Header size | 256 KiB |
| Encoded JPEG size | 16 MiB |
| Detections per frame | 512 |
| Detection string length | 256 characters |
| Image width/height | 4096 / 4096 |
| Image pixels | 16,000,000 |
| Connection read/handshake timeout | 5 seconds |
| Listener maintenance timeout | 1 second |
| Listen backlog | 5 |

Bounded environment overrides use the names `UAV_MAX_HEADER_SIZE`,
`UAV_MAX_JPEG_SIZE`, `UAV_MAX_DETECTIONS`, `UAV_MAX_STRING_LENGTH`,
`UAV_MAX_IMAGE_WIDTH`, `UAV_MAX_IMAGE_HEIGHT`, `UAV_MAX_IMAGE_PIXELS`,
`UAV_SOCKET_READ_TIMEOUT`, `UAV_LISTENER_TIMEOUT`, and `UAV_LISTEN_BACKLOG`.
Invalid or excessively weak/large values stop startup.

The bridge rejects malformed lengths before payload reads. It validates finite
JSON numbers, schema/type/range constraints, detection count, class/target text,
IDs, scores, timestamps, source dimensions, and bounding boxes. Unknown detection
keys are discarded. Boxes are clamped to the actual decoded image dimensions;
empty or reversed boxes reject the frame.

JPEG size and SOF dimensions are checked before OpenCV. Decode failure, dimension
mismatch, or excessive decoded dimensions reject the frame. No image or
detection message is published until all checks pass.

## Disconnects and errors

Idle or slow peers are disconnected after the read timeout, returning the bridge
to its bounded accept loop. TLS/authentication failures, unauthorized peer IPs,
invalid frames, and replay attempts are logged without secrets and are never
published. The legitimate sender reconnects with a fresh session ID after a
network failure.
