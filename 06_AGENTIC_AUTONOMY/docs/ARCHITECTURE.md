# Architecture

The CLI loads a JSON scenario and policy, validates both into frozen domain models, generates normalized tasks, evaluates every UAV candidate, validates all deterministic safety rules, and writes canonical JSON plus a readable console report.

Modules are separated into domain, loading, geometry, task generation, allocation/safety evaluation, orchestration, explanation, and serialization. Existing repository runtime code is neither imported nor modified.

Phase 2 adds state-history validation, lifecycle filtering, trigger generation, request-ID-based plan comparison, hysteresis selection, and replanning serialization. It calls the Phase 1 planner for every snapshot. The only Phase 1 extension is an optional candidate selector invoked after scoring and safety checks; omitting it preserves the Phase 1 path and byte output.
