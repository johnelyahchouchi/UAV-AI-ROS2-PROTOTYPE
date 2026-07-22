# Output schema

`schemas/mission_plan.schema.json` strictly describes the complete emitted plan. It defines tasks, approved assignments, rejected/unassigned tasks, all candidate evaluations, safety checks, reasons, score breakdowns, the summary, and the deterministic SHA-256 fingerprint. Required fields and allowed properties are explicit at every defined level. JSON keys and entity arrays have stable ordering.
