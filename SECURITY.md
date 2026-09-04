# Security Policy

## Reporting

Do not disclose credentials, private keys, private network details, or sensitive
model/data artifacts in a public issue. Prefer GitHub private vulnerability
reporting when it is enabled for this repository; otherwise contact the
maintainer privately and provide a minimal reproduction without real secrets.

Only the current default branch is maintained. This repository is a research
prototype, not a safety-certified or production flight-control system.

## Security assumptions

- The Windows sender and ROS 2 bridge use protocol version 2 over TLS 1.3.
- Mutual TLS is mandatory: each endpoint must trust the deployment CA and each
  endpoint must have its own certificate and private key.
- The bridge defaults to loopback and a loopback-only CIDR allowlist. Operators
  must explicitly configure both values for cross-machine deployment.
- TLS protects confidentiality and integrity in transit. Application session IDs
  and strictly increasing sequences provide bounded replay detection on top of
  the TLS session.
- Network JSON and JPEGs remain untrusted after authentication and are validated
  before ROS publication.
- ROS 2/DDS traffic is a separate trust boundary. Deployments must configure
  SROS2; TLS on the Windows bridge does not protect DDS traffic.

There is no insecure fallback. Missing certificates, an untrusted peer,
unsupported protocol versions, unknown checkpoint hashes, oversized inputs, or
invalid security environment variables fail closed.

## Model checkpoints

PyTorch `.pt` files can execute code during deserialization. Every local YOLO
checkpoint is SHA-256 verified against `00_PROJECT_GUIDE/ACTIVE_MODEL_HASHES.csv`
before Ultralytics is called. `UAV_TRUSTED_MODEL_REGISTRY` may name an external
reviewed registry. File names and absolute paths do not confer trust.

Do not commit model weights. Establish trust out of band, record the independently
verified SHA-256, store the artifact in access-controlled model storage, and only
then add its hash to the registry through review. Runtime auto-downloads are not
permitted.

## Secrets

- Never commit or log TLS private keys, tokens, PSKs, `.env` files, SROS2 private
  keys, or generated keystores.
- Provision secrets through operating-system protected files and environment
  variables with least-privilege filesystem permissions.
- Use separate sender and bridge identities and rotate both after suspected
  compromise.
- Test certificates must be generated ephemerally and deleted with the test
  workspace.

## Deployment requirements

1. Provision a private CA plus distinct bridge/server and sender/client
   certificates with correct DNS/IP subject alternative names.
2. Configure the environment variables documented in
   `07_DOCUMENTATION/03_TCP_PROTOCOL.md`.
3. Restrict host firewalls to expected peers in addition to the application CIDR
   allowlist.
4. Configure SROS2 according to `07_DOCUMENTATION/11_SROS2_DEPLOYMENT.md`.
5. Run the security tests, dependency audit, and static scan before deployment.

Bandit's subprocess warnings B404/B603/B607 are excluded in CI because the one
approved subprocess boundary uses a fixed `yt-dlp` executable, list-form
arguments, an option terminator, no shell, and a timeout. Changes to that boundary
require security review. CI fails on medium- and high-severity findings; low-level
exception-handling warnings remain visible during local full-severity scans but
do not fail the pipeline.
