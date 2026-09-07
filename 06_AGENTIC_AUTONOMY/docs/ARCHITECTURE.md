# Architecture

The CLI loads a JSON scenario and policy, validates both into frozen domain models, generates normalized tasks, evaluates every UAV candidate, validates all deterministic safety rules, and writes canonical JSON plus a readable console report.

Modules are separated into domain, loading, geometry, task generation, allocation/safety evaluation, orchestration, explanation, and serialization. Existing repository runtime code is neither imported nor modified.

Phase 2 adds state-history validation, lifecycle filtering, trigger generation, request-ID-based plan comparison, hysteresis selection, and replanning serialization. It calls the Phase 1 planner for every snapshot. The only Phase 1 extension is an optional candidate selector invoked after scoring and safety checks; omitting it preserves the Phase 1 path and byte output.

Phase 3 adds an adapter around those unchanged deterministic cores:

```text
canonical event stream / optional ROS message mappers
                         |
                         v
             strict normalized event parser
                         |
                         v
       serialized state store + freshness evaluator
                         |
                         v
      deterministic Phase 2 snapshot construction
                         |
                         v
             existing Phase 2 replanner
                         |
                         v
 snapshots + diagnostics + advisory replanning only
```

The event parser, state store, snapshot builder, file CLI, and duck-typed message
mappers have no ROS dependency. `ros_node.py` imports ROS packages only inside its
`main()` function and creates no ROS resources when imported. It owns a single
queue and assigns a monotonic shell ingestion sequence before the core applies an
event. Source timestamps and producer sequence values remain separate metadata.

The adapter never publishes commands. It has no velocity, trajectory, waypoint,
attitude, actuator, MAVROS, PX4, or ArduPilot interface.
