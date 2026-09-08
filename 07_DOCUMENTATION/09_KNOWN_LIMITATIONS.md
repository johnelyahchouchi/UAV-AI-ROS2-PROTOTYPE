# 09 - Known Limitations

## Overview

The current implementation demonstrates the feasibility of integrating artificial intelligence with a ROS 2-based robotic system for real-time perception.

However, the project represents an engineering prototype and several components can be further improved to increase robustness, scalability and operational performance.

The following limitations identify areas for future development rather than implementation errors.

---

## Threat Evaluation

The current threat evaluation module is based on predefined heuristic rules.

While suitable for demonstrating the architecture, the assigned threat levels have not been validated using operational data or quantitative performance metrics.

Future work could incorporate learned decision models or multi-sensor information to improve the reliability of threat assessment.

---

## Object Tracking

The system relies on BoT-SORT to maintain object identities across consecutive frames.

Although tracking performs well under normal operating conditions, temporary occlusions, rapid object motion or crowded scenes may still result in identity switches or lost tracks.

Future improvements may include stronger appearance models or sensor fusion techniques.

---
## Dataset Coverage

The performance of the detector is directly influenced by the diversity of the training dataset.

Objects, viewing angles or environmental conditions that are insufficiently represented during training may reduce detection accuracy in real-world deployments.

Expanding the dataset with additional environments and target variations would improve model generalization.

---

## System Scalability

The current architecture has been designed and validated as a prototype for a limited number of video streams.

Deployments involving multiple UAVs or higher data rates may require distributed processing, optimized communication protocols and additional computational resources to maintain real-time performance.

---

## Future Improvements

The modular architecture of the project allows individual components to be upgraded independently.

Future developments may include improved detection models, advanced threat assessment, additional sensors, optimized communication mechanisms and tighter integration with autonomous robotic behaviors.

---

## Security and deployment limitations

- Protocol version 2 protects the Windows-to-bridge hop only. DDS traffic needs a
  separately provisioned SROS2 keystore and enforce-mode deployment.
- Mutual-TLS certificates, private keys, firewall rules, and model artifacts are
  operator-managed and intentionally absent from Git.
- The in-process replay cache is bounded and is not persisted across a bridge
  restart. TLS still rejects ciphertext from old TLS sessions, but a compromised
  authenticated endpoint remains inside the trust boundary.
- OpenCV processes attacker-influenced JPEGs after encoded-size and header
  dimension checks. Keep the pinned supported OpenCV build patched and isolate
  the bridge host from unrelated workloads.
- Checkpoint allowlisting establishes file identity, not model quality or safety.
- Historical Git commits still contain model binaries and workstation inventory
  until a coordinated manual history rewrite is performed.
- The ROS 2 mirror is not the live Ubuntu workspace. Hardened bridge/shared files
  must be deployed to the real workspace together and validated there.
- Model weights, datasets, videos, TLS identities, and SROS2 keystores are external;
  a fresh checkout cannot run real inference or a deployed bridge without them.
- V1 reports robustness to the configured mild input perturbations only. Matching can
  split/merge targets in crowded scenes, pixel variation depends on resolution, and
  conditional confidence/class/localization statistics must be read with persistence.
- The `STABLE`, `INPUT-SENSITIVE`, and `UNSTABLE / REVIEW` bands are presentation
  thresholds for observed persistence, not accuracy or probability estimates.
- The V2 MC Dropout runtime and the registered external checkpoint have been manually
  validated locally. A fresh checkout still has no weights: V2 remains unavailable until
  `UAV_MCDO_V2_MODEL_PATH` points to that separately stored, hash-matching artifact.
  Repeated deterministic calls or V1 weights are not substitutes.
- MC Dropout output is an approximate model/epistemic uncertainty probe, not a calibrated
  correctness probability. Greedy IoU clustering can split or merge nearby objects.
- The bridge retains a temporary configurable BTR label rewrite for compatibility. It
  should be removed only after deployment consumers are audited.
