from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import nbformat
from nbformat.v4 import new_notebook

from agent import AgentRunResult, AgentTraceStep, AgentUsageSummary
from task_loader import count_data_source_types, serialize_task

if TYPE_CHECKING:
    from app_config import AppConfig

DEFAULT_RUNS_DIR = Path("jobs")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def bootstrap_notebook(notebook_path: Path) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = new_notebook(cells=[])
    with notebook_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)


def build_run_directory(*, model: str, current_time: datetime | None = None) -> Path:
    timestamp = (current_time or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_model_name = sanitize_model_name(model)
    return DEFAULT_RUNS_DIR / f"agent_{safe_model_name}_{timestamp}"


def sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
    return sanitized or "model"


def build_config_payload(config: AppConfig) -> dict[str, object]:
    return {
        "trial_name": config.run_dir.name,
        "trials_dir": str(config.run_dir.parent),
        "job_id": config.run_id,
        "agent": {
            "provider": "openrouter",
            "model_name": config.openrouter_model,
            "task_max_steps": config.max_steps,
            "temperature": 0.0,
        },
        "tasks": {
            "count": len(config.task_files),
            "task_files": [str(task_file.path) for task_file in config.task_files],
            "data_source_types": count_data_source_types([task_file.task for task_file in config.task_files]),
            "definitions": [serialize_task(task_file.task) for task_file in config.task_files],
        },
        "artifacts": {
            "run_dir": str(config.run_dir),
            "notebook_path": str(config.notebook_path),
            "transcript_path": str(config.transcript_path),
            "result_path": str(config.result_path),
            "log_path": str(config.log_path),
            "exception_path": str(config.exception_path),
            "task_artifacts_dir": str(config.task_artifacts_dir),
            "trajectory": "per-task in tasks/<stage>/trajectory.json",
        },
    }


def build_result_payload(
    *,
    config: AppConfig,
    stage_results: list[tuple[str, AgentRunResult]],
    task_results: list[TaskExecutionRecordLike],
    started_at: str,
    finished_at: str,
    exception_info: dict[str, str] | None,
) -> dict[str, object]:
    usage = summarize_usage(stage_results)
    all_trace_steps = flatten_trace_steps(stage_results)
    api_request_times = [round(step.metrics.api_duration_ms, 3) for step in all_trace_steps]

    return {
        "id": config.run_id,
        "trial_name": config.run_dir.name,
        "trial_uri": config.run_dir.resolve().as_uri(),
        "config": build_config_payload(config),
        "agent_result": {
            "n_input_tokens": usage.prompt_tokens,
            "n_output_tokens": usage.completion_tokens,
            "n_total_tokens": usage.total_tokens,
            "cost_usd": round(usage.cost_usd, 6),
            "metadata": {
                "n_stages": len(stage_results),
                "n_steps": len(all_trace_steps),
                "api_request_times_msec": api_request_times,
            },
        },
        "tasks": {
            "n_tasks": len(task_results),
            "n_completed_tasks": sum(1 for record in task_results if record.error_type is None),
            "n_failed_tasks": sum(1 for record in task_results if record.error_type is not None),
            "data_source_types": count_data_source_types([record.task for record in task_results]),
            "results": [
                {
                    **serialize_task(record.task),
                    "task_file": str(record.task_file_path),
                    "stage_name": record.stage_name,
                    "final_response": record.result.final_response,
                    "steps_used": record.result.steps_used,
                    "usage": asdict(record.result.usage),
                    "task_notebook": str(record.task_notebook_path.relative_to(config.run_dir)),
                    "status": "failed" if record.error_type is not None else "completed",
                    "failure_type": record.error_type,
                    "failure_message": record.error_message,
                }
                for record in task_results
            ],
        },
        "stages": [
            {
                "name": stage_name,
                "steps_used": result.steps_used,
                "final_response": result.final_response,
                "usage": asdict(result.usage),
            }
            for stage_name, result in stage_results
        ],
        "exception_info": exception_info,
        "started_at": started_at,
        "finished_at": finished_at,
        "artifacts": {
            "transcript": str(config.transcript_path.relative_to(config.run_dir)),
            "trajectory": "per-task in tasks/<stage>/trajectory.json",
            "runtime_log": str(config.log_path.relative_to(config.run_dir)),
        },
    }


def build_trajectory_payload(
    *,
    config: AppConfig,
    stage_results: list[tuple[str, AgentRunResult]],
) -> dict[str, object]:
    return {
        "schema_version": "dsbench-run-v1",
        "session_id": config.run_id,
        "agent": {
            "provider": "openrouter",
            "model_name": config.openrouter_model,
            "extra": {
                "temperature": 0.0,
                "task_max_steps": config.max_steps,
            },
        },
        "steps": [serialize_trace_step(step) for step in flatten_trace_steps(stage_results)],
    }


def flatten_trace_steps(stage_results: list[tuple[str, AgentRunResult]]) -> list[AgentTraceStep]:
    return [step for _, result in stage_results for step in result.trace_steps]


def summarize_usage(stage_results: list[tuple[str, AgentRunResult]]) -> AgentUsageSummary:
    return AgentUsageSummary(
        prompt_tokens=sum(result.usage.prompt_tokens for _, result in stage_results),
        completion_tokens=sum(result.usage.completion_tokens for _, result in stage_results),
        total_tokens=sum(result.usage.total_tokens for _, result in stage_results),
        cost_usd=sum(result.usage.cost_usd for _, result in stage_results),
    )


def serialize_trace_step(step: AgentTraceStep) -> dict[str, object]:
    return {
        "step_id": step.step_id,
        "stage": step.stage,
        "timestamp": step.timestamp,
        "source": "assistant",
        "request_messages": list(step.request_messages),
        "message": step.assistant_content,
        "tool_calls": [
            {
                "tool_call_id": tool_call.tool_call_id,
                "function_name": tool_call.name,
                "arguments_json": tool_call.arguments_json,
            }
            for tool_call in step.tool_calls
        ],
        "observation": {"results": [{"content": tool_result} for tool_result in step.tool_results]},
        "metrics": asdict(step.metrics),
    }


def write_transcript(
    transcript_path: Path,
    *,
    config: AppConfig,
    stage_results: list[tuple[str, AgentRunResult]],
) -> None:
    lines = [
        f"trial_name: {config.run_dir.name}",
        f"model_name: {config.openrouter_model}",
        f"task_count: {len(config.task_files)}",
        f"run_dir: {config.run_dir}",
        "",
    ]

    for stage_name, result in stage_results:
        lines.append(f"=== Stage: {stage_name} ===")
        lines.append(f"final_response: {result.final_response}")
        lines.append(f"steps_used: {result.steps_used}")
        lines.append(
            "usage: "
            f"prompt_tokens={result.usage.prompt_tokens}, "
            f"completion_tokens={result.usage.completion_tokens}, "
            f"total_tokens={result.usage.total_tokens}, "
            f"cost_usd={result.usage.cost_usd:.6f}"
        )
        lines.append("")

        for step in result.trace_steps:
            lines.append(
                f"--- Step {step.step_id} | {step.timestamp} | "
                f"api_duration_ms={step.metrics.api_duration_ms:.3f} ---"
            )
            lines.append("REQUEST MESSAGES:")
            lines.append(json.dumps(list(step.request_messages), indent=2))
            lines.append("ASSISTANT CONTENT:")
            lines.append(step.assistant_content or "<empty>")

            if step.tool_calls:
                lines.append("TOOL CALLS:")
                for tool_call in step.tool_calls:
                    lines.append(
                        json.dumps(
                            {
                                "tool_call_id": tool_call.tool_call_id,
                                "name": tool_call.name,
                                "arguments_json": tool_call.arguments_json,
                            },
                            indent=2,
                        )
                    )

            if step.tool_results:
                lines.append("TOOL RESULTS:")
                for index, tool_result in enumerate(step.tool_results, start=1):
                    lines.append(f"[tool_result_{index}]")
                    lines.append(tool_result)

            lines.append("")

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def persist_task_notebook(
    *,
    source_notebook_path: Path,
    task_artifacts_dir: Path,
    stage_name: str,
) -> Path:
    task_dir = task_artifacts_dir / stage_name
    task_dir.mkdir(parents=True, exist_ok=True)
    destination = task_dir / "notebook.ipynb"
    shutil.copy2(source_notebook_path, destination)
    return destination


def persist_task_trajectory(
    *,
    task_artifacts_dir: Path,
    stage_name: str,
    config: "AppConfig",
    stage_results: list[tuple[str, AgentRunResult]],
) -> Path:
    """Write trajectory.json for a single task alongside its notebook."""
    trajectory_path = task_artifacts_dir / stage_name / "trajectory.json"
    write_json(trajectory_path, build_trajectory_payload(config=config, stage_results=stage_results))
    return trajectory_path


class TaskExecutionRecordLike:
    task: object
    task_file_path: Path
    stage_name: str
    result: AgentRunResult
    task_notebook_path: Path
    task_trajectory_path: Path
    error_type: str | None
    error_message: str | None
