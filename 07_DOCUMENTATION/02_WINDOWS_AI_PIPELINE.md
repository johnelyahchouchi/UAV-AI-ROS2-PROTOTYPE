# 02 - Windows AI Pipeline

## Active applications

- `win_yolo_tcp_sender_botsort_threat.py`: operational Windows perception sender.
- `live_screen_model_tester.py`: thin local live/video tester entry point.
- `uav_ai_control_panel.py`: optional GUI configuration for the active sender.
- `model_test_dashboard/`: recorded-video batch analysis and export UI.

The baseline sender is an intentional protected regression copy, not another entry
point. Tooling such as demo-video creation and model smoke tests lives under
`01_WINDOWS_AI/tools/`.

## Sender path

```text
configured source
  -> OpenCV frame
  -> Ultralytics YOLO + BoT-SORT
  -> existing target/threat calculation
  -> bounded JPEG and validated detection metadata
  -> protocol v2 framing inside mutual TLS 1.3
```

The launcher requires an explicit external trusted model and approved source. Host,
port, TLS identity paths, thresholds, stride, and send width are CLI/environment
configuration; no company VM address or workstation path is embedded.

## Local live tester path

Normal playback decodes/captures one raw frame, runs exactly one YOLO prediction, then
draws filtered boxes, confidence, FPS, and latency. Pressing `U` copies the unannotated
current frame, pauses capture, and invokes the separate V1 robustness adapter. Extra
predictions never run in the normal frame loop.

V1 measures response to deterministic mild input perturbations. It reports persistence,
confidence variation, class agreement/entropy/evidence share, center and size variation,
and reference IoU separately. It is not Bayesian uncertainty, correctness probability,
or an accuracy estimate. The V2 runtime holds the input fixed while six validated
late-head dropout layers vary internal model state over 20 lower-level forwards. V2 is
selectable only when its separate external checkpoint passes trust and architecture
checks; that artifact is not distributed in source Git.

See `LIVE_SCREEN_MODEL_TESTER.md` for the canonical no-ROS presentation workflow.
