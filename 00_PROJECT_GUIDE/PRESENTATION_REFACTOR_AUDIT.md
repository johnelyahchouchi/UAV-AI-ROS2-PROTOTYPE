# Presentation Refactor Audit

Audit date: 2026-09-04  
Working branch: `refactor/presentation-ready-integration`  
Source baseline: `main` at `1d7be430091f91f9b37ff9fb8bc72dd9d367275f`

This is the reviewed cleanup plan created before repository files were removed. The
checked-out repository, imports, launchers, tests, documentation, workflows, and all
Git branches were inspected. Names such as `v1`, `v5`, and `BASELINE` were not treated
as deletion evidence by themselves.

## Findings and classification

| Area | Classification | Decision |
|---|---|---|
| `uav_security/`, security tests and security documentation | KEEP | These are current trust boundaries for models, archives, transport, images, and CSV output. |
| Active Windows sender and its baseline copy | KEEP | `AGENTS.md` explicitly protects the baseline. The two files are byte-equivalent and remain as an intentional regression reference. |
| `01_WINDOWS_AI/apps/live_screen_model_tester.py` | KEEP + REFACTOR | Preserve the CLI while moving reusable code into `01_WINDOWS_AI/live_tester/` and adding an on-demand uncertainty boundary. |
| `01_WINDOWS_AI/model_test_dashboard/` | KEEP | This is the recorded-video batch test/export UI. It is distinct from the low-latency live tester. |
| Sender launcher family | MERGE / CONSOLIDATE | Keep one parameterized PowerShell launcher and one BAT wrapper. Remove interactive and experiment-specific wrappers after updating references. |
| Standalone remote-stream demo | DELETE | It had import-time side effects, fixed CUDA/classes, and a hardcoded public URL. The maintained sender/control panel supports configured URL sources; the live tester covers local screen/video demos. |
| BTR demo-video builder | KEEP + MOVE | Retain the useful generator under `01_WINDOWS_AI/tools/media/`; it is a tool, not an active application. |
| Live-tester launchers | CONSOLIDATE | Keep the direct BAT launcher requested for double-click presentation use; it forwards all CLI options and avoids PowerShell policy dependence. Remove the incomplete, unreferenced PowerShell duplicate. |
| Legacy archive-review PowerShell script | DELETE | It targets a former Desktop tree and is a one-off migration tool. The durable rationale belongs in documentation, not an executable mover. |
| Three ROS dashboard scripts | KEEP + REFACTOR | They provide operational imagery, aggregate analytics, and event timeline views. Give them unversioned filenames, retain their topic behavior, and document the roles. |
| `04_DATASET_ENGINEERING/` code and lineage documents | KEEP | Current code uses repository-relative defaults or environment variables; datasets remain external. |
| `05_TRAINING/**/args.yaml` and `results.csv` | KEEP | Lightweight experiment configuration and metrics preserve reproducibility. |
| `05_TRAINING/**/*.png` and `05_TRAINING/**/*.jpg` | GENERATED — SHOULD BE GITIGNORED | Ultralytics regenerates these plots and batch previews. No operational code or documentation embeds them. |
| Model checkpoints, datasets, and test videos | EXTERNAL ASSET | No model or media binary is tracked on current `main`; registries, hashes, and provenance remain in source. |
| `08_MODEL_UNCERTAINTY/outputs/` | GENERATED — SHOULD BE GITIGNORED | The tracked JSON/CSV contain machine-specific absolute paths. Replace them with documented schemas and deterministic tests. |
| `feature/model-uncertainty-dashboard-v1` core | MERGE / CONSOLIDATE | Selectively port perturbations, class-agnostic matching, transparent metrics, and in-memory analysis. Do not merge the old Gradio application or unsafe direct model loader. |
| MC Dropout V2 | ARCHITECTURE ONLY | No validated V2 implementation exists in any branch. Provide an explicit extension contract and unavailable status; never simulate it with deterministic repeats. |
| `06_AGENTIC_AUTONOMY/` | KEEP | It is a separate simulation-first subsystem with its own tests and packaging. No shared helper change requires refactoring it. |
| Root and `07_DOCUMENTATION/` narrative | KEEP + REFACTOR | Replace the legacy root stub with a fast project tour and update live demo, uncertainty, ROS dashboard, and test instructions. |
| Generated workstation inventory CSVs in `00_PROJECT_GUIDE/` | DELETE | They describe former trees and now contradict the repository. Preserve current registries and replace cleanup knowledge with this reviewed audit. |
| `.pytest_cache`, `__pycache__`, local outputs and local ignored weights | GENERATED / EXTERNAL | Keep ignored and do not delete operator-owned local artifacts as part of source cleanup. |

## Uncertainty decision

The older engine is scientifically V1 input-perturbation robustness: one clean
inference plus seeded brightness, contrast, blur, noise, and JPEG variants. Its
transparent persistence, confidence, class-agreement/entropy, bounding-box variation,
and IoU metrics are useful and will be ported.

Repository-wide Git history searches found no executable V2 implementation that keeps
the full model and BatchNorm layers in evaluation mode while activating only
`Dropout2d` for repeated inference on one exact frame. V2 therefore remains an explicit
adapter contract pending a dropout-capable checkpoint and validated implementation.

## Safety checks performed before cleanup

- Compared the active sender and baseline copy.
- Searched application imports, subprocess calls, tests, launchers, documentation,
  ROS/SROS2 references, README content, and GitHub workflows.
- Confirmed current Git tracks no `.pt`, `.pth`, `.onnx`, `.engine`, or video files.
- Confirmed tracked files above 250 KiB are generated training images only.
- Confirmed training images are referenced only by generated inventory reports.
- Confirmed uncertainty outputs are runtime results and expose local absolute paths.
- Confirmed the dashboard scripts subscribe to the current ROS topics but serve three
  distinct views rather than version copies of the same view.

## Intended presentation architecture

```text
local image/video or Windows screen
        -> trusted YOLO detector
        -> recorded test dashboard or live tester
        -> on-demand V1 robustness inspector (U)
        -> transparent per-target metrics and interpretation

Windows sender -> mutually authenticated TCP -> ROS 2 bridge
                                      -> operational / analytics / timeline dashboards

V2 MC Dropout -> extension contract only until validated model support exists
```

## V2 integration addendum (2026-09-06)

The classification above records the evidence available during the original cleanup.
A subsequently supplied, validated Colab notebook is now retained under
`08_MODEL_UNCERTAINTY/notebooks/` and its six-layer MC Dropout runtime has been extracted
into reviewed source modules. The external trained checkpoint was not found locally, so
V2 remains disabled by default and no trust-registry digest was invented. When the exact
checkpoint is obtained, the runtime requires SHA-256 allowlisting, verifies all six
`Dropout2d(p=0.20)` placements, keeps the model/BatchNorm in evaluation mode, and exposes
V2 only as an on-demand same-frame method. The original cleanup rationale remains an
accurate historical audit rather than being silently rewritten.

## Applied deletion inventory

The final local diff removes 232 tracked paths. No history rewrite was performed.

### Generated training images (197 files, 47,622,804 bytes)

All `.jpg`/`.png` files in the following run folders were Ultralytics plots, curves,
confusion matrices, label previews, and train/validation batch previews. No source,
launcher, test, workflow, or documentation embeds them. `args.yaml` and `results.csv`
remain for reproducibility, and root ignore rules prevent regeneration from being added.

| Run folder | Removed files |
|---|---:|
| `classification_runs/armored_vehicle_classifier_v1` | 15 |
| `classification_runs/artillery_launcher_classifier_v1` | 15 |
| `classification_runs/tank_platform_classifier_v0` | 15 |
| `classification_runs/tank_platform_classifier_v1_exact_types` | 15 |
| `classification_runs/tank_platform_classifier_v2_exact_focus` | 15 |
| `classification_runs/tank_type_classifier_v3_only_tanks` | 15 |
| `classification_runs/tank_type_classifier_v3_only_tanks-2` | 15 |
| `classification_runs/tank_type_classifier_v4_safe_unknown` | 15 |
| `detection_runs/amad5_aerial_yolov8s_v1` | 20 |
| `detection_runs/btr_yolov8n_local_test` | 17 |
| `detection_runs/btr_yolov8n_v2_50epochs` | 20 |
| `detection_runs/military_kaggle_yolov8s_v1` | 20 |

### Generated/stale audit files (21 files)

The following workstation snapshots were point-in-time inventories of former source,
dataset, model, output, and training trees. They had no runtime consumer and contained
stale/machine-specific claims. Current registries plus this reviewed audit replace them:

`ACTIVE_LEGACY_PATH_REFERENCES.csv`, `BASELINE_PREFLIGHT_CHECK.csv`,
`CLEAN_PROJECT_INVENTORY.csv`, `DATASET_PATH_MIGRATION_REPORT.csv`,
`DATASET_PATH_REFERENCES.csv`, `DATASET_SCRIPT_INVENTORY.csv`,
`DETECTOR_DATASET_COPY_AUDIT.csv`, `DETECTOR_DATASET_VALIDATION.csv`,
`LEGACY_DATASET_PATH_REFERENCES.csv`, `MODEL_EXACT_DUPLICATES.csv`,
`MODEL_FILE_INVENTORY.csv`, `MODEL_HASH_INVENTORY.csv`, `OLD_CODE_CANDIDATES.csv`,
`OLD_SENDER_COMPARISON.csv`, `ORIGINAL_SOURCE_INVENTORY_SANITIZED.csv`,
`OUTPUT_FILE_INVENTORY.csv`, `ROS2_WINDOWS_MIRROR_SOURCE.txt`,
`TRAINING_DATA_PATH_AUDIT.csv`, `TRAINING_RUN_ARGS_SUMMARY.csv`,
`TRAINING_RUN_FOLDER_SIZES.csv`, and `UNCLASSIFIED_ROOT_PYTHON_FILES.csv`.

### Applications and launchers (8 files)

- `apps/live_yolo_stream.py`: unreferenced fixed-URL/fixed-CUDA demo, replaced by the
  configured sender/control-panel URL path and local live tester.
- `apps/make_btr_demo_video.py`: moved to `tools/media/make_btr_demo_video.py`.
- `launchers/02_Review_And_Archive_Old_UAV_Files.ps1`: one-off legacy-tree mover;
  durable decisions are captured here.
- `launchers/Start_Clean_Baseline.ps1`: superseded setup wrapper.
- `launchers/start_btr_phase1_conf45.ps1`: experiment-specific sender wrapper,
  replaced by parameters on `start_yolo_sender.ps1`.
- `launchers/start_live_screen_tester.ps1`: unreferenced incomplete duplicate,
  replaced by the argument-forwarding presentation BAT.
- `launchers/start_yolo_sender_full_menu.ps1` and `start_yolo_sender_menu.ps1`:
  duplicate interactive wrappers replaced by the canonical sender PS1/BAT pair.

### Dashboard replacements (3 files)

- `uav_clean_target_dashboard_v5.py` -> `uav_operational_dashboard.py`
- `uav_analytics_dashboard_v2.py` -> `uav_analytics_dashboard.py`
- `uav_tank_type_timeline_dashboard_v1.py` -> `uav_timeline_dashboard.py`

Only versioned filenames and matching node/window identifiers changed; topic behavior
remains covered by static tests and documented SROS2 enclave migration.

### Other generated/runtime records (3 files)

- `05_TRAINING/results_summary/RUN_LOCATIONS.txt`: stale machine run locations,
  replaced by portable training documentation and retained per-run metadata.
- `08_MODEL_UNCERTAINTY/outputs/tank_video_10s/summary.json` and `targets.csv`:
  generated results containing local input/output paths, replaced by deterministic
  schema-level tests and the uncertainty README.
