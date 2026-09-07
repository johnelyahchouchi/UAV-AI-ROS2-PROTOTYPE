# Phase 1 limitations

This planner uses declared state and local 2D points. It does not estimate energy consumption, generate routes or search patterns, model altitude/terrain/weather/collisions, replan in real time, command aircraft, use ROS 2/PX4, process imagery/event cameras, or call an LLM. Polygon self-intersection is not detected. JSON Schema is published as a contract but runtime validation is implemented without a third-party schema library.

Phase 2 replanning is snapshot-driven rather than real-time. Its distances and movement thresholds are abstract scenario units, not GPS or meters. Hysteresis defaults are research parameters. Return-home and landing records are advisory and may be impossible to deliver after link loss.

Phase 3 does not add live flight control, custom ROS interfaces, ament packaging,
TF2, coordinate conversion, target fusion, pixel-to-world projection, persistent
checkpointing, or bounded-history rollover. The prototype keeps its full Phase 2
history until a configured snapshot or byte limit is reached, then fails clearly
instead of truncating it. Persistent checkpoints and an explicitly designed
bounded-history format are future work.

The adapter initially accepts only exact `mission_local` planning coordinates.
Legacy YOLO detections without world coordinates cannot become projectable planner
targets. DDS arrival order is not inherently deterministic: repeatability applies
to an identical normalized, ordered event stream after the ROS shell assigns its
ingestion sequence.
