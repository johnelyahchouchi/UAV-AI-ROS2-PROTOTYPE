# Phase 2 — Dynamic Mission Replanning

Phase 2 processes a strictly ordered file of complete mission-state snapshots. It remains pure Python, deterministic, offline, and independent of ROS 2, PX4, perception, networking, GPUs, and LLM decision logic.

## Snapshot contract

Each snapshot contains the complete Phase 1 scenario plus one lifecycle record for every mission request. `sequence` is the sole ordering authority. An optional timestamp is preserved as an opaque label and never generated or used for ordering. Requests may be introduced as `PENDING` or `ACTIVE`; requests may not disappear. `COMPLETED` and `CANCELLED` are terminal.

The stable identity of a task is its mission-request ID. Phase 1 task IDs are still deterministic, but their sequence prefix can change when a higher-priority task is introduced. Plan comparison therefore never relies on the Phase 1 task number.

## Replanning procedure

For each snapshot the system validates history and lifecycle transitions, detects state triggers, filters to active tasks, invokes the Phase 1 planner, applies the configured continuity selector only among candidates already approved by Phase 1 safety, compares the new plan with the previous plan, and emits decisions and advisory return-home records.

Phase 1 remains the source of truth for scoring, capability checks, battery reserve, link quality, availability, distance, workload, region safety, target continuity, and deterministic candidate ordering. An unsafe incumbent is never retained.

## Hysteresis

The default `minimum_base_score_improvement_to_switch` is 75 points. This is a configurable research default, not a validated safety value. A challenger whose improvement is exactly 75 points may replace the incumbent. The comparison uses `base_total`, because the task-priority multiplier is identical for every UAV candidate for the same task.

If the incumbent remains safe, it is retained until the best safe challenger meets the threshold. If the incumbent fails any safety rule, it is immediately replaced by the highest-ranked safe candidate or the task becomes unassigned.

## Movement and units

The target-movement trigger defaults to 100 scenario units. These are abstract local Cartesian scenario units, not GPS distance or meters. Target movement can cause a score change, but reassignment still requires either an unsafe incumbent or the hysteresis threshold.

## Workload meaning

`current_workload` in every snapshot is external workload that was already active before planning and is not represented by the snapshot's mission requests. The Phase 1 allocator begins with that external value and increments workload as it allocates the current active requests in deterministic task order.

Do not copy assignments from the previous generated plan into `current_workload`; doing so would count those assignments twice. If external workload changes independently, update the snapshot value and the replanner will emit `UAV_WORKLOAD_CHANGED`.

## Battery and link advisories

Battery below the Phase 1 reserve produces `BATTERY_BELOW_RESERVE` and a return-home recommendation. Battery below `critical_battery_percent` instead produces `BATTERY_CRITICAL` and an urgent safe-return-or-landing recommendation.

Link-based return-home advice is controlled by `return_home_on_link_below_minimum`. When enabled, the output explicitly warns that the recommendation may not be executable because communication may already be degraded or lost. All records are advisory JSON only. Phase 2 never generates a flight command, route, waypoint, or control message.

## Canonical fingerprints

Fingerprinting recursively removes every object property named `deterministic_fingerprint`, including nested Phase 1 plans and snapshot records. The remaining value is serialized using the shared canonical JSON procedure: UTF-8, sorted keys, two-space indentation, finite JSON numbers, and a final newline. SHA-256 is calculated over those exact UTF-8 bytes.

Snapshot fingerprints are calculated from their snapshot record. The history fingerprint is calculated from the complete result after recursively excluding all fingerprint fields. This avoids circular hashing and makes changes to fingerprint strings alone irrelevant to the canonical digest.

## Change types

- `UNCHANGED`: the same request remains with the same UAV.
- `NEW_ASSIGNMENT`: a new, activated, or formerly unassigned task receives a UAV.
- `REASSIGNED`: ownership transfers between UAVs.
- `COMPLETED`: a terminal completed task leaves active planning.
- `CANCELLED`: a terminal cancelled task leaves active planning.
- `NEWLY_UNASSIGNED`: a previously assigned active task has no safe candidate.
- `REMAINS_UNASSIGNED`: a pending or already-unassigned task remains without an assignment.

`NO_MEANINGFUL_CHANGE` is emitted only when no other meaningful state, lifecycle, safety, or hysteresis trigger exists.
