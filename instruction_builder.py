from __future__ import annotations

from task_loader import BenchmarkTask


_INSTRUCTION_REWRITES: tuple[tuple[str, str], ...] = (
    ("notebook code", "workspace code, scripts, or notebooks"),
    ("directly in notebook code", "directly in the workspace"),
    ("use notebook code", "use code and tools in the workspace"),
    ("notebook", "workspace"),
)


def build_task_instruction(task: BenchmarkTask) -> str:
    sections = [
        "# Task",
        task.problem_statement,
        "",
        "# Question",
        task.question,
        "",
        "# Workspace",
        "- The task data is available under `/workspace/data`.",
        "- You may use shell commands, Python scripts, SQL engines, or notebooks inside the container.",
        "- Do not modify the source data under `/workspace/data`.",
        "- Write the final answer to `/workspace/final_answer.txt`.",
        "- `/workspace/final_answer.txt` must contain only the answer, with no explanation or extra prose.",
        "- Optional supporting artifacts may be written to `/logs/artifacts/` for later inspection.",
    ]

    extra_guidance = _rewrite_agent_instructions(task.agent_instructions)
    if extra_guidance:
        sections.extend(["", "# Additional Guidance", extra_guidance])

    return "\n".join(sections).strip() + "\n"


def _rewrite_agent_instructions(instructions: str) -> str:
    rewritten = instructions.strip()
    if not rewritten:
        return ""

    for old, new in _INSTRUCTION_REWRITES:
        rewritten = rewritten.replace(old, new)

    return rewritten
