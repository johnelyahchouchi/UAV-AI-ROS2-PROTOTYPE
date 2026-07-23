# Allocation policy

All adjustable weights, thresholds, priority multipliers, and score scale live in `config/default_policy.json`.

Each component is normalized to the configured score scale. Capability is binary. Battery and link scores measure headroom above their safety thresholds. Distance decreases linearly to zero at the normalization distance. Workload decreases with used capacity. Continuity is binary. The weighted base score is multiplied by task priority.

With the default score scale, every component and `base_total` is in the inclusive range `0..1000`. `final_total` applies the configured priority multiplier and is therefore intentionally allowed to exceed 1000. With the default multipliers its range is `0..1500`. The multiplier is common to all UAV candidates for a task, so it preserves their ranking; it expresses the priority-adjusted value of the result for reporting and future cross-task use.

Policy loading requires the exact documented keys. Weights are nonnegative integers totaling 100, multipliers and the score scale are positive integers, and every numeric value must be finite. Battery reserve must be in `[0, 100)`, and minimum link quality must be in `[0, 1)` so normalized headroom remains well-defined.

Candidates are ranked by final score, continuity, capability, battery, distance, link, workload, then lexicographically by UAV ID. Every candidate receives reasons, a full score breakdown, and safety checks. Only candidates passing all checks can be selected.

The `replanning` policy section contains the Phase 2 switching margin, target-movement threshold in scenario units, critical-battery threshold, and configurable link-based return-home behavior. The 75-point switching margin and 100-scenario-unit movement threshold are research defaults, not validated operational or safety values.
