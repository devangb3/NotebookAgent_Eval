from __future__ import annotations

from pathlib import Path

from task_loader import BenchmarkTask


def build_task_prompt(task: BenchmarkTask, *, data_root: Path, max_steps: int) -> str:
    resolved_data_source = task.resolved_data_source_path(data_root)
    instruction_block = ""
    if task.agent_instructions:
        instruction_block = f"Task-specific instructions:\n{task.agent_instructions.strip()}\n\n"

    return (
        "Starting from a blank notebook, act as a data analyst who must use python notebook code "
        "execution before answering.\n"
        f"You have at most {max_steps} steps to complete this task.\n\n"
        f"{instruction_block}"
        "Task Format:\n"
        f"1. Data Source: {task.data_source_type}\n"
        f"   Path: `{resolved_data_source}`\n"
        f"2. Problem Statement: {task.problem_statement}\n"
        f"3. Question: {task.question}\n\n"
        "Requirements:\n"
        "- Inspect the data source in notebook code before answering.\n"
        "- Load only the minimum data needed to answer the question.\n"
        "- Keep the notebook deterministic and concise.\n"
        f"{_data_source_guidance(task.data_source_type)}\n"
        "- Show intermediate analysis in notebook cells when it materially supports the answer.\n"
        "- Finish with a concise plain-text answer and do not call more tools after that.\n"
    )


def _data_source_guidance(data_source_type: str) -> str:
    if data_source_type == "csv":
        return "- Treat the source as one or more CSV files and use pandas to inspect, aggregate, and analyze them."
    if data_source_type == "table":
        return "- Treat the source as structured tabular data. If the path is a directory, inspect the relevant tables/files before loading only the needed parts."
    if data_source_type == "images":
        return "- Inspect image files programmatically using available Python libraries. If a directory is provided, sample only what is needed to answer the question."
    if data_source_type == "text":
        return "- Read the relevant text files in notebook code, extract the evidence needed for the question, and cite the evidence in the final answer."
    return "- Use notebook code to inspect the declared data source and answer the question."
