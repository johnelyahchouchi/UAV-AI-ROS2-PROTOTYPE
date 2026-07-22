def console_report(plan: dict, output_path: str) -> str:
    lines = [f"Mission: {plan['scenario_id']}", f"Generated tasks: {plan['summary']['generated_task_count']}",
             f"Assigned: {plan['summary']['assigned_task_count']}", f"Unassigned: {plan['summary']['unassigned_task_count']}", ""]
    task_map = {x["id"]: x for x in plan["tasks"]}
    for assignment in plan["assignments"]:
        task = task_map[assignment["task_id"]]
        lines.extend([f"[APPROVED] {assignment['task_id']} -> {assignment['uav_id']}",
                      f"  Type: {task['task_type']}", f"  Priority: {task['priority']}",
                      f"  Score: {assignment['score_breakdown']['final_total']}", "  Why:"])
        lines.extend(f"    - {reason}" for reason in assignment["reasons"])
    for item in plan["unassigned_tasks"]:
        lines.extend([f"[REJECTED] {item['task_id']} -> unassigned", f"  Why: {item['reasons'][0]}"])
    lines.extend(["", f"Fingerprint: {plan['deterministic_fingerprint']}", f"Output: {output_path}"])
    return "\n".join(lines)
