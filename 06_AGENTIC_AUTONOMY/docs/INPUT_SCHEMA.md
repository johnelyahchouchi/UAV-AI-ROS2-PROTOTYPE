# Input schema

`schemas/mission_scenario.schema.json` is the formal contract. The standard-library runtime boundary mirrors its strict behavior: it rejects malformed JSON, duplicate or unknown properties, unsupported schema versions, missing fields, invalid enums, numeric strings, booleans used as numbers, non-finite values, empty identifiers, duplicate IDs, invalid ranges, empty fleets, invalid geometry, and broken references.

All current task types are entity-based. `INVESTIGATE_TARGET` and `TRACK_TARGET` require exactly one `target_id` and no `region_id`. `SEARCH_REGION`, `OBSERVE_REGION`, and `RELAY_COMMUNICATIONS` require exactly one `region_id` and no `target_id`.

Coordinates are finite local 2D Cartesian values in scenario-consistent units. `max_task_distance`, when present, must be finite and greater than zero.
