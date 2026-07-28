# Phase 1 limitations

This planner uses declared state and local 2D points. It does not estimate energy consumption, generate routes or search patterns, model altitude/terrain/weather/collisions, replan in real time, command aircraft, use ROS 2/PX4, process imagery/event cameras, or call an LLM. Polygon self-intersection is not detected. JSON Schema is published as a contract but runtime validation is implemented without a third-party schema library.

Phase 2 replanning is snapshot-driven rather than real-time. Its distances and movement thresholds are abstract scenario units, not GPS or meters. Hysteresis defaults are research parameters. Return-home and landing records are advisory and may be impossible to deliver after link loss.
