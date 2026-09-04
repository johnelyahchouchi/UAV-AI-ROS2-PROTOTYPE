# Security Hardening Report

Date: 2026-09-04

Branch: `security/repository-hardening`

Base commit: `15cf943`

No push, merge, release, remote-branch change, history rewrite, force operation,
working-tree cleanup, or model/dataset deletion was performed.

## Repository findings

- The working tree was initially clean on `main`; this work was moved to the
  dedicated local `security/repository-hardening` branch before edits.
- The active sender and its baseline copy were byte-identical before the change
  and remain byte-identical after the security integration.
- The checked-out `03_MODELS/active` location referenced by guide/dashboard
  defaults does not exist. The retained training-run checkpoints are local
  artifacts, not a suitable public model store. Launchers now require an explicit
  external `UAV_MODEL_PATH`.
- Bare `python` on this Windows host is Amesim Python 2.7.12. CPython 3.11 is
  available through `py -3.11`; a separate controlled YOLO environment contains
  NumPy 1.26.4, OpenCV 4.10.0, Ultralytics 8.4.107, and the dashboard dependencies.
- The repository is only a mirror of the Ubuntu ROS 2 side; ROS 2, PX4, GPU
  access, cameras, model deployment storage, certificates, and an SROS2 keystore
  cannot be assumed on this computer.
- `RUNBOOK.md`, `MODEL_REGISTRY.md`, and `DATASET_REGISTRY.md` contained malformed
  control characters and, in the model guide, an embedded PowerShell fragment.
  They were repaired while preserving their intended operational information.
- The repository did not have a root dependency policy, security CI, Dependabot,
  SROS2 deployment guidance, or a security policy. Those controls are now added.

## A. Security finding verification

1. **Confirmed — unauthenticated plaintext frame transport.** The active sender,
   baseline sender, and ROS 2 bridge used bare TCP; the bridge broadly bound and
   published peer-supplied data. They now deploy together as protocol v2 over
   TLS 1.3 with mutual certificate verification, ALPN `uav-frame/2`, mandatory
   external certificate files, a loopback bind default, and CIDR filtering before
   TLS/frame processing. There is no plaintext fallback.
2. **Confirmed — unbounded framing/memory growth.** Peer-controlled header and
   JPEG sizes were not capped and `recv_exact` repeatedly concatenated bytes.
   Central limits now reject zero, invalid, or oversized values before payload
   reads; a bounded `bytearray` receive loop handles EOF, TLS failure, and timeout.
3. **Confirmed — idle-client socket denial of service.** The bridge used blocking
   single-connection reads. Listener and connection timeouts, a bounded backlog,
   clean disconnect handling, and sender reconnects return the bridge to accept
   after an idle/failed client. The one-client design is retained intentionally.
4. **Confirmed — detection JSON was insufficiently validated.** A central strict
   schema now validates the frame envelope and each detection, rejects non-finite
   or excessive values/depth/text/counts, discards unknown fields, canonicalizes
   geometry, and clamps boxes to the successfully decoded image dimensions before
   any ROS publication.
5. **Confirmed — sequence/replay checks were absent.** Each TLS connection now
   claims a fresh 256-bit application session ID and requires strictly increasing
   63-bit sequence numbers. Duplicates, older values, session changes, and reuse of
   a recently claimed session on a new TLS connection are rejected.
6. **Confirmed — JPEG decoding lacked resource validation.** Encoded size and JPEG
   SOF dimensions are checked before OpenCV. Width, height, pixel count, decoded
   channel count, and header/decoded dimension equality are enforced afterward.
7. **Confirmed — `.pt` checks did not establish trust.** All practical YOLO load
   paths now calculate SHA-256 and fail closed unless the digest appears in the
   explicit trusted registry. Registry paths and names are ignored as trust data;
   Ultralytics is invoked only after verification.
8. **Confirmed — model weights were tracked publicly.** Twenty-four `.pt` files
   were tracked. They are staged for removal from Git tracking, remain present as
   ignored local artifacts, and all 24 local copies were verified after the index
   change. No tracked `.pth`, `.onnx`, or `.engine` existed. The active best-model
   hashes each still have one preserved local matching copy. Historical removal is
   a separate, approval-gated procedure.
9. **Confirmed — CSV formula injection.** CSV exports containing model-, network-,
   source-, or user-derived strings now pass through one helper that prefixes
   strings beginning with `=`, `+`, `-`, `@`, tab, or carriage return. Numeric
   cells and ordinary labels are unchanged.
10. **Confirmed — external ZIP handling was unsafe/unbounded.** The one
    `extractall` call and direct unbounded metadata/image reads were replaced with
    reusable pre-validation and bounded reads/extraction. Path traversal, absolute
    paths, duplicates, symlink/device entries, encryption, member/aggregate sizes,
    member count, metadata size, and compression ratio are checked.
11. **Partly confirmed; command injection was not reproducible.** The control panel
    already used list-form `subprocess.Popen` without a shell, so the claimed shell
    injection was not present. Its inputs were insufficiently validated. IP, port,
    confidence, IoU, image size, stride, width, and binary flags are now validated
    before argv construction; PowerShell uses argument arrays or native arguments.
12. **Confirmed — YouTube substring classification.** All duplicate logic now uses
    exact parsed HTTP(S) host allowlisting. `yt-dlp` uses list argv, `--`, a timeout,
    checked exit status, and empty-output/error handling.
13. **Confirmed — internal IP/path portability issues.** The old VM IP and
    workstation-specific operational paths were removed. Runtime paths derive from
    the repository or explicit environment variables. Tracked historical reports
    retain logical information with workstation and dataset roots redacted.
14. **Confirmed — workstation inventory privacy exposure.** The 38.5 MB original
    inventory held 196,011 absolute workstation paths and timestamps describing
    about 54.6 GB of local files. It is staged out of tracking and ignored while its
    local copy is preserved. A small aggregate replacement removes paths,
    usernames, and timestamps; related tracked inventories have redacted roots.
15. **Confirmed — dependency/supply-chain controls were incomplete.** Security-
    sensitive and component dependencies are pinned, the verified Ultralytics
    `8.4.107` line is preserved, and Dependabot plus least-privilege CI now run
    compile, tests, `pip-audit`, Bandit, and private-key-pattern checks. `SECURITY.md`
    documents trust boundaries, checkpoint risk, secrets, and deployment.

Additional requested controls were also addressed. Implicit YOLO base-model
downloads were removed; training requires an explicit local checkpoint that is
already trusted. No SROS2 deployment existed, so an operator runbook now documents
enforce mode, keystore expectations, and per-node enclaves without committing
keys. Security limits are centralized and bounded environment overrides fail
closed. Logs no longer expose full model paths or credential/query-bearing stream
URLs.

## B. Files changed

New shared security code:

- `uav_security/__init__.py`
- `uav_security/config.py`
- `uav_security/csv_safe.py`
- `uav_security/detection.py`
- `uav_security/image_validation.py`
- `uav_security/input_validation.py`
- `uav_security/model_integrity.py`
- `uav_security/safe_zip.py`
- `uav_security/source_urls.py`
- `uav_security/transport.py`

New security tests:

- `tests/security/conftest.py`
- `tests/security/test_config.py`
- `tests/security/test_csv_safe.py`
- `tests/security/test_detection.py`
- `tests/security/test_image_validation.py`
- `tests/security/test_input_validation.py`
- `tests/security/test_model_integrity.py`
- `tests/security/test_safe_zip.py`
- `tests/security/test_source_urls.py`
- `tests/security/test_tls_integration.py`
- `tests/security/test_transport.py`

Repository controls and documentation:

- `.gitignore`
- `.github/dependabot.yml`
- `.github/workflows/security-ci.yml`
- `requirements-security-ci.txt`
- `requirements-windows.txt`
- `SECURITY.md`
- `README.md`
- `07_DOCUMENTATION/03_TCP_PROTOCOL.md`
- `07_DOCUMENTATION/09_KNOWN_LIMITATIONS.md`
- `07_DOCUMENTATION/11_SROS2_DEPLOYMENT.md`
- `07_DOCUMENTATION/12_MANUAL_HISTORY_CLEANUP.md`
- `07_DOCUMENTATION/13_SECURITY_HARDENING_REPORT.md`
- `06_AGENTIC_AUTONOMY/README.md`
- `06_AGENTIC_AUTONOMY/pyproject.toml`

Transport, launch, dashboard, and model-loading integrations:

- `01_WINDOWS_AI/apps/live_yolo_stream.py`
- `01_WINDOWS_AI/apps/make_btr_demo_video.py`
- `01_WINDOWS_AI/apps/uav_ai_control_panel.py`
- `01_WINDOWS_AI/apps/win_yolo_tcp_sender_botsort_threat.py`
- `01_WINDOWS_AI/apps/win_yolo_tcp_sender_botsort_threat_BASELINE.py`
- `01_WINDOWS_AI/launchers/Start_Clean_Baseline.ps1`
- `01_WINDOWS_AI/launchers/Start_UAV_AI_Control_Panel.bat`
- `01_WINDOWS_AI/launchers/Start_UAV_Windows_Sender.bat`
- `01_WINDOWS_AI/launchers/start_btr_phase1_conf45.ps1`
- `01_WINDOWS_AI/launchers/start_yolo_sender.ps1`
- `01_WINDOWS_AI/launchers/start_yolo_sender_full_menu.ps1`
- `01_WINDOWS_AI/launchers/start_yolo_sender_menu.ps1`
- `01_WINDOWS_AI/model_test_dashboard/README.md`
- `01_WINDOWS_AI/model_test_dashboard/launch_dashboard.ps1`
- `01_WINDOWS_AI/model_test_dashboard/requirements.txt`
- `01_WINDOWS_AI/model_test_dashboard/requirements-dev.txt`
- `01_WINDOWS_AI/model_test_dashboard/src/uav_model_dashboard/configuration.py`
- `01_WINDOWS_AI/model_test_dashboard/src/uav_model_dashboard/model_manager.py`
- `01_WINDOWS_AI/model_test_dashboard/src/uav_model_dashboard/video_processor.py`
- `01_WINDOWS_AI/model_test_dashboard/tests/conftest.py`
- `01_WINDOWS_AI/model_test_dashboard/tests/test_configuration.py`
- `01_WINDOWS_AI/model_test_dashboard/tests/test_model_manager.py`
- `01_WINDOWS_AI/model_test_dashboard/tests/test_video_processor.py`
- `01_WINDOWS_AI/tools/data_export/win_yolo_data_extractor.py`
- `01_WINDOWS_AI/tools/smoke_tests/test_all_active_models.py`
- `01_WINDOWS_AI/tools/tracking/clean_target_tracker.py`
- `02_ROS2_WINDOWS_MIRROR/bridge/uav_windows_tcp_frame_bridge.py`
- `02_ROS2_WINDOWS_MIRROR/dashboards/uav_analytics_dashboard_v2.py`
- `02_ROS2_WINDOWS_MIRROR/dashboards/uav_clean_target_dashboard_v5.py`
- `02_ROS2_WINDOWS_MIRROR/dashboards/uav_tank_type_timeline_dashboard_v1.py`

Dataset and training integrations:

- `04_DATASET_ENGINEERING/builders/build_armored_vehicle_classifier_dataset_v1.py`
- `04_DATASET_ENGINEERING/builders/build_artillery_launcher_classifier_dataset_v1.py`
- `04_DATASET_ENGINEERING/builders/build_tank_platform_classifier_dataset.py`
- `04_DATASET_ENGINEERING/builders/build_tank_platform_classifier_dataset_v2_exact_focus.py`
- `04_DATASET_ENGINEERING/builders/build_tank_type_classifier_dataset_v3.py`
- `04_DATASET_ENGINEERING/builders/build_tank_type_classifier_dataset_v4_safe_unknown.py`
- `04_DATASET_ENGINEERING/builders/create_artillery_launcher_folders_v1.py`
- `04_DATASET_ENGINEERING/cleaners/clean_amad5_filenames.py`
- `04_DATASET_ENGINEERING/cleaners/clean_tank_dataset.py`
- `04_DATASET_ENGINEERING/cleaners/select_useful_frames_v3.py`
- `04_DATASET_ENGINEERING/cleaners/sort_artillery_images_v1.py`
- `04_DATASET_ENGINEERING/cleaners/sort_tank_platform_crops.py`
- `04_DATASET_ENGINEERING/importers/extract_frames_v3.py`
- `04_DATASET_ENGINEERING/importers/extract_tank_platform_crops.py`
- `04_DATASET_ENGINEERING/importers/import_artillery_launcher_labeled_zips_v1.py`
- `04_DATASET_ENGINEERING/importers/import_exact_tank_type_images.py`
- `04_DATASET_ENGINEERING/importers/import_roboflow_detection_to_platform_crops.py`
- `04_DATASET_ENGINEERING/inspectors/audit_tank_platform_dataset.py`
- `04_DATASET_ENGINEERING/inspectors/dataset_inventory.py`
- `04_DATASET_ENGINEERING/inspectors/inspect_all_artillery_zips.py`
- `04_DATASET_ENGINEERING/inspectors/inspect_artillery_keywords_in_zip.py`
- `04_DATASET_ENGINEERING/inspectors/inspect_yolo_zip_classes.py`
- `04_DATASET_ENGINEERING/inventory/DETECTOR_DATASET_VALIDATION_SUMMARY.md`
- `05_TRAINING/scripts/train_btr_local.py`
- `05_TRAINING/scripts/train_btr_v2.py`
- `05_TRAINING/scripts/train_kaggle_military_v1.py`
- `05_TRAINING/configs/detection/README.md`
- `05_TRAINING/configs/detection/original_training_configs/military_kaggle_original.yaml`
- `05_TRAINING/results_summary/RUN_LOCATIONS.txt`
- `05_TRAINING/classification_runs/armored_vehicle_classifier_v1/args.yaml`
- `05_TRAINING/classification_runs/artillery_launcher_classifier_v1/args.yaml`
- `05_TRAINING/classification_runs/tank_platform_classifier_v0/args.yaml`
- `05_TRAINING/classification_runs/tank_platform_classifier_v1_exact_types/args.yaml`
- `05_TRAINING/classification_runs/tank_platform_classifier_v2_exact_focus/args.yaml`
- `05_TRAINING/classification_runs/tank_type_classifier_v3_only_tanks/args.yaml`
- `05_TRAINING/classification_runs/tank_type_classifier_v3_only_tanks-2/args.yaml`
- `05_TRAINING/classification_runs/tank_type_classifier_v4_safe_unknown/args.yaml`
- `05_TRAINING/detection_runs/amad5_aerial_yolov8s_v1/args.yaml`
- `05_TRAINING/detection_runs/btr_yolov8n_local_test/args.yaml`
- `05_TRAINING/detection_runs/btr_yolov8n_v2_50epochs/args.yaml`
- `05_TRAINING/detection_runs/military_kaggle_yolov8s_v1/args.yaml`

The `args.yaml` edits above redact workstation paths only; training
hyperparameters are unchanged.

Guide/inventory changes:

- `00_PROJECT_GUIDE/ACTIVE_LEGACY_PATH_REFERENCES.csv`
- `00_PROJECT_GUIDE/ACTIVE_MODEL_HASHES.csv`
- `00_PROJECT_GUIDE/ACTIVE_SENDER_SELECTED.txt`
- `00_PROJECT_GUIDE/BASELINE_PREFLIGHT_CHECK.csv`
- `00_PROJECT_GUIDE/CLEAN_PROJECT_INVENTORY.csv`
- `00_PROJECT_GUIDE/DATASET_PATH_MIGRATION_REPORT.csv`
- `00_PROJECT_GUIDE/DATASET_PATH_REFERENCES.csv`
- `00_PROJECT_GUIDE/DATASET_REGISTRY.md`
- `00_PROJECT_GUIDE/DATASET_SCRIPT_INVENTORY.csv`
- `00_PROJECT_GUIDE/DETECTOR_DATASET_VALIDATION.csv`
- `00_PROJECT_GUIDE/LEGACY_DATASET_PATH_REFERENCES.csv`
- `00_PROJECT_GUIDE/MODEL_FILE_INVENTORY.csv`
- `00_PROJECT_GUIDE/MODEL_HASH_INVENTORY.csv`
- `00_PROJECT_GUIDE/MODEL_REGISTRY.md`
- `00_PROJECT_GUIDE/OLD_CODE_CANDIDATES.csv`
- `00_PROJECT_GUIDE/ROS2_WINDOWS_MIRROR_SOURCE.txt`
- `00_PROJECT_GUIDE/RUNBOOK.md`
- `00_PROJECT_GUIDE/TRAINING_DATA_PATH_AUDIT.csv`
- `00_PROJECT_GUIDE/TRAINING_RUN_ARGS_SUMMARY.csv`
- `00_PROJECT_GUIDE/TRAINING_RUN_FOLDER_SIZES.csv`
- `00_PROJECT_GUIDE/UNCLASSIFIED_ROOT_PYTHON_FILES.csv`
- Added `00_PROJECT_GUIDE/ORIGINAL_SOURCE_INVENTORY_SANITIZED.csv`
- Staged removal from tracking of
  `00_PROJECT_GUIDE/ORIGINAL_SOURCE_INVENTORY.csv`; the ignored local file remains

Model files staged out of tracking, but retained locally and ignored:

- `05_TRAINING/classification_runs/armored_vehicle_classifier_v1/weights/best.pt`
- `05_TRAINING/classification_runs/armored_vehicle_classifier_v1/weights/last.pt`
- `05_TRAINING/classification_runs/artillery_launcher_classifier_v1/weights/best.pt`
- `05_TRAINING/classification_runs/artillery_launcher_classifier_v1/weights/last.pt`
- `05_TRAINING/classification_runs/tank_platform_classifier_v0/weights/best.pt`
- `05_TRAINING/classification_runs/tank_platform_classifier_v0/weights/last.pt`
- `05_TRAINING/classification_runs/tank_platform_classifier_v1_exact_types/weights/best.pt`
- `05_TRAINING/classification_runs/tank_platform_classifier_v1_exact_types/weights/last.pt`
- `05_TRAINING/classification_runs/tank_platform_classifier_v2_exact_focus/weights/best.pt`
- `05_TRAINING/classification_runs/tank_platform_classifier_v2_exact_focus/weights/last.pt`
- `05_TRAINING/classification_runs/tank_type_classifier_v3_only_tanks/weights/best.pt`
- `05_TRAINING/classification_runs/tank_type_classifier_v3_only_tanks/weights/last.pt`
- `05_TRAINING/classification_runs/tank_type_classifier_v3_only_tanks-2/weights/best.pt`
- `05_TRAINING/classification_runs/tank_type_classifier_v3_only_tanks-2/weights/last.pt`
- `05_TRAINING/classification_runs/tank_type_classifier_v4_safe_unknown/weights/best.pt`
- `05_TRAINING/classification_runs/tank_type_classifier_v4_safe_unknown/weights/last.pt`
- `05_TRAINING/detection_runs/amad5_aerial_yolov8s_v1/weights/best.pt`
- `05_TRAINING/detection_runs/amad5_aerial_yolov8s_v1/weights/last.pt`
- `05_TRAINING/detection_runs/btr_yolov8n_local_test/weights/best.pt`
- `05_TRAINING/detection_runs/btr_yolov8n_local_test/weights/last.pt`
- `05_TRAINING/detection_runs/btr_yolov8n_v2_50epochs/weights/best.pt`
- `05_TRAINING/detection_runs/btr_yolov8n_v2_50epochs/weights/last.pt`
- `05_TRAINING/detection_runs/military_kaggle_yolov8s_v1/weights/best.pt`
- `05_TRAINING/detection_runs/military_kaggle_yolov8s_v1/weights/last.pt`

## C. Security architecture

The sender validates its settings, verifies its local model hash, requires client
certificate/key/CA files, and opens a TLS 1.3 connection whose bridge certificate
name and CA chain are checked. The bridge checks the peer IP allowlist before the
TLS handshake, requires a trusted client certificate, and checks the negotiated
TLS version and ALPN. TLS record AEAD supplies confidentiality and integrity.

Inside TLS, a four-byte bounded header length precedes strict JSON and a bounded
JPEG. A new random session ID is bound into every header; the bridge claims it for
one TLS connection and accepts only increasing sequences. It then validates JPEG
structure/dimensions, decodes safely, sanitizes the allowlisted detection schema,
and only then updates publishable state. Any failed stage drops the connection or
frame before ROS publication.

`.pt` loading is a separate trust boundary. `load_trusted_yolo` hashes the complete
local file, compares it in constant time against SHA-256 values from the trusted
registry, and invokes Ultralytics only after a match. This identifies an approved
artifact; it does not certify model behavior.

Default limits are 256 KiB header, 16 MiB JPEG, 512 detections, 256-character
strings, 4096x4096 dimensions, 16 million pixels, five-second reads, one-second
listener maintenance, backlog five, 100,000 ZIP members, 2 GiB per archive member,
50 GiB total uncompressed archive data, 1000:1 compression ratio, and 1 MiB small
metadata reads. Environment overrides themselves have hard minimum/maximum bounds.

## D. Compatibility changes

Protocol v2 is intentionally incompatible with the plaintext protocol. Old sender
+ new bridge and new sender + old bridge fail closed. Deploy the active sender,
baseline sender, root `uav_security` package, and bridge together. ROS topic names,
YOLO inference/tracking, threat scoring, model classes, uncertainty calculations,
training hyperparameters, dashboard behavior, and Mission Copilot logic were not
redesigned.

The default bind/target is now loopback. Cross-machine use requires an explicit
bind address, CIDR allowlist, sender target, firewall rule, and certificates.
Launchers require an explicit verified Python executable and model artifact.

## E. Environment configuration

Required for the bridge:

- `UAV_BRIDGE_TLS_CERT`: bridge certificate chain file
- `UAV_BRIDGE_TLS_KEY`: bridge private-key file
- `UAV_BRIDGE_TLS_CA`: trusted deployment CA file

Required for the sender:

- `UAV_SENDER_TLS_CERT`: sender client certificate chain file
- `UAV_SENDER_TLS_KEY`: sender client private-key file
- `UAV_BRIDGE_TLS_CA`: trusted deployment CA file
- `UAV_YOLO_PYTHON`: verified Python executable for repository launchers
- `UAV_MODEL_PATH`: explicit trusted local model for standard launchers

Network settings:

- `UAV_BRIDGE_HOST`: sender target, default `127.0.0.1`
- `UAV_BRIDGE_BIND_ADDRESS`: bridge listen address, default `127.0.0.1`
- `UAV_BRIDGE_PORT`: default `5010`
- `UAV_BRIDGE_ALLOWED_CIDRS`: default `127.0.0.0/8,::1/128`
- `UAV_BRIDGE_TLS_SERVER_NAME`: expected server certificate identity, default
  sender target

Optional trust/limit settings:

- `UAV_TRUSTED_MODEL_REGISTRY`: external reviewed registry; default
  `00_PROJECT_GUIDE/ACTIVE_MODEL_HASHES.csv`
- `UAV_MAX_HEADER_SIZE`, `UAV_MAX_JPEG_SIZE`, `UAV_MAX_DETECTIONS`,
  `UAV_MAX_STRING_LENGTH`, `UAV_MAX_IMAGE_WIDTH`, `UAV_MAX_IMAGE_HEIGHT`,
  `UAV_MAX_IMAGE_PIXELS`, `UAV_SOCKET_READ_TIMEOUT`, `UAV_LISTENER_TIMEOUT`,
  `UAV_LISTEN_BACKLOG`, `UAV_MAX_ARCHIVE_MEMBERS`,
  `UAV_MAX_ARCHIVE_MEMBER_SIZE`, `UAV_MAX_ARCHIVE_SIZE`,
  `UAV_MAX_ARCHIVE_RATIO`, and `UAV_MAX_METADATA_SIZE`

Training additionally requires `UAV_BASE_MODEL_PATH` and
`UAV_TRAINING_DATA_YAML`; `UAV_TRAINING_OUTPUT_DIR` is optional. Base YOLO hashes
present only in historical inventory were not silently promoted to trusted status.

## F. Verification performed

- Initial existing Mission Copilot suite: **66 passed**.
- Final security regression suite in isolated Python 3.11 CI environment:
  **81 passed**.
- Final existing Mission Copilot suite in the same environment: **66 passed**.
- Existing model-dashboard suite in its controlled CUDA/Ultralytics environment:
  **28 passed**, with 13 upstream Matplotlib/Pyparsing deprecation warnings.
- Full targeted Python `compileall`: **passed**.
- Six modified PowerShell files parsed through the PowerShell AST parser:
  **passed**.
- Two GitHub YAML files parsed with PyYAML: **passed**.
- `pip-audit --strict` in the isolated pinned environment: **no known
  vulnerabilities found**.
- Bandit medium/high scan with documented subprocess exclusions: **passed**.
- Private-key/credential-pattern scan: **clean**.
- Sender and baseline copy after the security patch: **byte-identical**.
- Final duplicate audit: no wildcard bind, `extractall`, implicit literal `.pt`
  YOLO load, YouTube substring test, old VM IP, absolute user path, specific Unix
  home, quadratic receive concatenation, or shell-enabled subprocess; no model binary or cache,
  key, PEM, or keystore remains tracked in the index.

The TLS integration test generates an ephemeral CA, server identity, trusted
client identity, and untrusted client identity. It proves a real mutually
authenticated TLS 1.3 packet round trip and untrusted-client rejection. Other
tests cover CIDR policy, malformed/oversized/truncated/idle frames, replay,
strict JSON/detection/image rules, checkpoint hashes, formulas, ZIP attacks, URL
lookalikes, and control-panel validation.

## G. Remaining manual steps

1. Review this diff before staging anything else.
2. Back up the 24 preserved local checkpoints into approved access-controlled
   artifact storage; decide which represent company/IP assets and who may access
   them.
3. Independently establish trust for any base training checkpoint before adding
   its SHA-256 to the active or an external reviewed registry.
4. Provision a private CA and distinct server/client identities, protect private
   keys with OS permissions, set certificate environment variables, and restrict
   the host firewall.
5. Deploy the mirror changes into the real Ubuntu ROS 2 workspace and run a real
   sender-to-bridge-to-topic test.
6. Create/review the SROS2 keystore, governance, permissions, and enclaves outside
   Git; test enforce mode with an unenrolled node.
7. Only after policy review and explicit approval, follow
   `12_MANUAL_HISTORY_CLEANUP.md` in a fresh mirror clone. The destructive
   history commands have not been executed.

## H. Remaining risks

- TLS and SROS2 identity provisioning, key rotation, certificate expiry, firewall
  policy, and the live Ubuntu deployment remain operator responsibilities.
- The application replay cache is in-memory, bounded to 4096 recent sessions, and
  resets with the bridge process. TLS still prevents replay of captured ciphertext
  into another TLS session.
- An authenticated/allowlisted but compromised sender is inside the trust boundary
  and can consume resources up to the documented per-frame and archive limits.
- OpenCV's native JPEG decoder remains security-sensitive after pre-validation;
  keep the pinned supported version patched and isolate the bridge host.
- SHA-256 allowlisting identifies checkpoint bytes but does not prove provenance,
  quality, absence of model backdoors, or operational safety. A hostile local actor
  able to modify files during verification/loading is outside this prototype's
  assumed host-integrity boundary.
- Historical commits and forks retain model binaries and the original workstation
  inventory until coordinated cleanup. Treat exposed paths/timestamps as already
  public and rotate any real secret found separately.
- The full ROS 2/GPU/camera runtime was not available on this computer, so the
  real deployment needs the documented end-to-end acceptance test. This remains a
  research prototype, not a safety-certified flight-control system.

## I. Safe follow-up commands

Inspect the local change:

```powershell
git branch --show-current
git status --short
git diff --stat
git diff
git diff --cached --stat
git diff --cached
```

Reproduce security and Mission Copilot tests in an ignored virtual environment:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --requirement requirements-security-ci.txt
& .\.venv\Scripts\python.exe -m pytest -q tests\security
$env:PYTHONPATH = "$PWD\06_AGENTIC_AUTONOMY\src"
& .\.venv\Scripts\python.exe -m pytest -q 06_AGENTIC_AUTONOMY\tests
& .\.venv\Scripts\python.exe -m pip_audit --strict
& .\.venv\Scripts\python.exe -m bandit -q -ll -r uav_security 01_WINDOWS_AI\apps\win_yolo_tcp_sender_botsort_threat.py 01_WINDOWS_AI\apps\uav_ai_control_panel.py 02_ROS2_WINDOWS_MIRROR\bridge\uav_windows_tcp_frame_bridge.py 04_DATASET_ENGINEERING\importers -s B404,B603,B607
```

Configure already-provisioned local certificate files for one PowerShell session
without copying them into the repository:

```powershell
$SecurityDirectory = Join-Path $env:LOCALAPPDATA "UAVSecurity"
$env:UAV_BRIDGE_TLS_CA = Join-Path $SecurityDirectory "deployment-ca.crt"
$env:UAV_SENDER_TLS_CERT = Join-Path $SecurityDirectory "sender.crt"
$env:UAV_SENDER_TLS_KEY = Join-Path $SecurityDirectory "sender.key"
$env:UAV_BRIDGE_TLS_CERT = Join-Path $SecurityDirectory "bridge.crt"
$env:UAV_BRIDGE_TLS_KEY = Join-Path $SecurityDirectory "bridge.key"
$env:UAV_YOLO_PYTHON = Read-Host "Absolute verified Python executable"
$env:UAV_MODEL_PATH = Read-Host "Absolute trusted model checkpoint"
$env:UAV_BRIDGE_HOST = Read-Host "Bridge IP address"
$env:UAV_BRIDGE_TLS_SERVER_NAME = Read-Host "Bridge certificate DNS/IP identity"
```

On the Ubuntu bridge host, configure the corresponding environment and start the
deployed mirror only after substituting protected real paths and network policy:

```bash
export UAV_BRIDGE_TLS_CA=/etc/uav-security/deployment-ca.crt
export UAV_BRIDGE_TLS_CERT=/etc/uav-security/bridge.crt
export UAV_BRIDGE_TLS_KEY=/etc/uav-security/bridge.key
export UAV_BRIDGE_BIND_ADDRESS="$(hostname -I | awk '{print $1}')"
export UAV_BRIDGE_ALLOWED_CIDRS='replace-with-sender-cidr'
export UAV_BRIDGE_PORT=5010
python3 02_ROS2_WINDOWS_MIRROR/bridge/uav_windows_tcp_frame_bridge.py
```

Then start the Windows sender with the already configured environment:

```powershell
& $env:UAV_YOLO_PYTHON 01_WINDOWS_AI\apps\win_yolo_tcp_sender_botsort_threat.py `
    --target $env:UAV_BRIDGE_HOST `
    --port 5010 `
    --source (Read-Host "Video path, camera index, or approved URL") `
    --model $env:UAV_MODEL_PATH
```

Inspect ROS topics on the Ubuntu host after an authenticated connection:

```bash
ros2 topic echo /uav_1/coco_detections --once
ros2 topic hz /uav_1/camera/image_raw
```

After review, stage without `git add .`, inspect again, and create a local commit:

```powershell
git add -u
git add .github .gitignore SECURITY.md requirements-security-ci.txt requirements-windows.txt tests uav_security 00_PROJECT_GUIDE\ORIGINAL_SOURCE_INVENTORY_SANITIZED.csv 07_DOCUMENTATION\11_SROS2_DEPLOYMENT.md 07_DOCUMENTATION\12_MANUAL_HISTORY_CLEANUP.md 07_DOCUMENTATION\13_SECURITY_HARDENING_REPORT.md
git status --short
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "security: harden repository trust boundaries"
```

Do not push or merge until the reviewed local commit is approved.
