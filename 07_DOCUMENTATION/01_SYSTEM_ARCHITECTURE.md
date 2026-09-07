# 01 - System Architecture

The repository contains complementary research paths rather than one mandatory runtime.

```text
operator-provided video/screen
        -> trusted YOLO load (SHA-256 allowlist)
        -> live tester: one prediction per normal frame
             -> U: frozen-frame V1 input-perturbation robustness

video/camera source -> YOLO + BoT-SORT + threat fields
        -> authenticated protocol v2 over mutual TLS 1.3
        -> ROS 2 bridge mirror
        -> operational / analytics / timeline dashboards

mission JSON -> deterministic Mission Copilot -> explainable simulated recommendations
```

## Windows AI

The active sender detects, tracks, computes existing threat metadata, and sends bounded
frames/detections. The live tester and recorded model dashboard are local evaluation
tools; they do not publish ROS topics or command an aircraft. Model files are external
artifacts and are verified before deserialization.

## ROS 2 side

`02_ROS2_WINDOWS_MIRROR/` mirrors deployable bridge/dashboard source. It is not a ROS 2
workspace and does not prove that ROS 2, SROS2 identities, or the Ubuntu deployment are
installed on this Windows machine. The bridge rejects plaintext and invalid protocol-v2
messages before publishing the existing topics.

## Mission Copilot

`06_AGENTIC_AUTONOMY/` is a standard-library, simulation-first deterministic core. Its
recommendations remain separated from ROS/PX4 adapters and do not control motors,
actuators, weapons, or harmful actions.

## Trust and deployment boundaries

- Model checkpoints, datasets, videos, TLS keys/certificates, and SROS2 keystores stay
  outside source Git.
- Transport details and required environment variables are in `03_TCP_PROTOCOL.md`.
- SROS2 deployment is separately described in `11_SROS2_DEPLOYMENT.md`.
- This prototype is not safety-certified flight software.
