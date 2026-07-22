# Architecture

The CLI loads a JSON scenario and policy, validates both into frozen domain models, generates normalized tasks, evaluates every UAV candidate, validates all deterministic safety rules, and writes canonical JSON plus a readable console report.

Modules are separated into domain, loading, geometry, task generation, allocation/safety evaluation, orchestration, explanation, and serialization. Existing repository runtime code is neither imported nor modified.

