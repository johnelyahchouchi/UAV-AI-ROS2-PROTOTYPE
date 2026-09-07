# 10 - Development Roadmap

## Implemented foundation

- trusted YOLO model loading, Windows detection/tracking, and existing threat metadata;
- authenticated bounded Windows-to-ROS protocol v2;
- local recorded and live/video test paths;
- on-demand V1 input-perturbation robustness inspection;
- validated V2 MC Dropout runtime extraction and on-demand selector, gated by an
  external trusted checkpoint;
- three role-based ROS dashboard views;
- deterministic simulation-first Mission Copilot recommendations and replanning.

## Near-term validation

1. Deploy the renamed bridge/dashboard nodes to the real Ubuntu workspace and regenerate
   matching SROS2 enclaves.
2. Run the live presentation checklist with the external allowlisted model on the target
   GPU and record representative clear/difficult/negative-control observations.
3. Keep the lightweight live, uncertainty, dashboard, and security CI green while
   adding optional artifact-driven GPU/model smoke tests outside normal source Git.
4. Remove the bridge's temporary BTR label-rewrite option only after confirming deployment
   consumers and preserving required behavior.

## Research extensions

- Validate V1 thresholds across curated datasets and resolutions.
- Retrieve and independently hash the exact Colab V2 checkpoint, enroll it in the trust
  registry, verify its six forward-connected dropout placements, and complete the real
  GPU/video smoke workflow without changing the validated architecture.
- Add future ROS 2/event-camera adapters around, rather than inside, the deterministic
  Mission Copilot core.
- Evaluate tracking under occlusion, multi-UAV scaling, sensor fusion, and deployment
  acceleration without changing the current safety boundaries.

Nothing in this roadmap implies autonomous flight authorization or safety certification.
