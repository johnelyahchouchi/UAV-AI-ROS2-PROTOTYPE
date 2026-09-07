# Windows Applications

- `win_yolo_tcp_sender_botsort_threat.py` is the active tracked-perception sender.
- `live_screen_model_tester.py` is the thin entry point for the reusable live/video
  tester and on-demand V1 robustness inspector.
- `uav_ai_control_panel.py` is an optional GUI for configuring the active sender.
- `win_yolo_tcp_sender_botsort_threat_BASELINE.py` is a protected, byte-equivalent
  regression reference required by `AGENTS.md`; it is not an alternative launcher.

Recorded batch testing lives in `../model_test_dashboard/`. Media generation, smoke
tests, tracking helpers, and data export live under `../tools/`.
