from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import traceback
from uuid import uuid4

import nbformat
from dotenv import load_dotenv
from nbformat.v4 import new_notebook
from openai import OpenAI

from agent import (
    AgentConfig,
    AgentRunResult,
    AgentTraceStep,
    AgentUsageSummary,
    NotebookReActAgent,
)
from environment import NotebookEnvironment
from headroom_tasks import HEADROOM_TASKS, HeadroomTask, serialize_headroom_task, task_stage_name
from tools import NotebookToolExecutor

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_RUNS_DIR = Path("jobs")
PHASE1_MAX_STEPS = 50
PHASE2_MAX_STEPS = 50
NOTEBOOK_TIMEOUT_SECONDS = 600
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AppConfig:
    openrouter_api_key: str
    openrouter_model: str
    home_credit_data_dir: Path
    notebook_timeout_seconds: int
    run_id: str
    run_dir: Path
    notebook_path: Path
    transcript_path: Path
    trajectory_path: Path
    config_path: Path
    result_path: Path
    log_path: Path


def main() -> None:
    load_dotenv()
    config = load_config()
    configure_logging(config.log_path)
    started_at = utc_now()
    logger.info(
        "Starting run %s | model=%s | data_dir=%s | notebook_timeout_seconds=%s",
        config.run_id,
        config.openrouter_model,
        config.home_credit_data_dir,
        config.notebook_timeout_seconds,
    )
    client = OpenAI(api_key=config.openrouter_api_key, base_url=OPENROUTER_BASE_URL)
    write_json(
        config.config_path,
        build_config_payload(config),
    )

    bootstrap_notebook(notebook_path=config.notebook_path)

    stage_results: list[tuple[str, AgentRunResult]] = []
    phase1_result: AgentRunResult | None = None
    phase2_task_results: list[tuple[HeadroomTask, AgentRunResult]] = []
    exception_info: dict[str, str] | None = None
    phase1_prompt = build_phase1_prompt(config.home_credit_data_dir)

    try:
        with NotebookEnvironment(
            config.notebook_path,
            timeout_seconds=config.notebook_timeout_seconds,
        ) as environment:
            tools = NotebookToolExecutor(environment)
            phase1_agent = NotebookReActAgent(
                client=client,
                tools=tools,
                config=AgentConfig(model=config.openrouter_model, max_steps=PHASE1_MAX_STEPS),
            )
            phase2_agent = NotebookReActAgent(
                client=client,
                tools=tools,
                config=AgentConfig(model=config.openrouter_model, max_steps=PHASE2_MAX_STEPS),
            )

            phase1_result = phase1_agent.run(phase1_prompt, stage_name="phase1")
            stage_results.append(("phase1", phase1_result))
            logger.info("Phase 1 completed in %s steps", phase1_result.steps_used)
            print("Phase 1 result:")
            print(phase1_result.final_response)
            print()

            for index, task in enumerate(HEADROOM_TASKS, start=1):
                stage_name = task_stage_name(task)
                phase2_prompt = build_phase2_prompt(config.home_credit_data_dir, task)
                logger.info(
                    "Starting phase 2 task %s/%s: %s (%s)",
                    index,
                    len(HEADROOM_TASKS),
                    task.task_id,
                    task.stage,
                )
                print(f"Phase 2 [{index}/{len(HEADROOM_TASKS)}] {task.task_id} — {task.stage}")
                task_result = phase2_agent.run(phase2_prompt, stage_name=stage_name)
                stage_results.append((stage_name, task_result))
                phase2_task_results.append((task, task_result))
                logger.info(
                    "Completed phase 2 task %s in %s steps",
                    task.task_id,
                    task_result.steps_used,
                )
                print(task_result.final_response)
                print()
    except Exception as exc:
        exception_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        logger.exception("Run failed")
        raise
    finally:
        write_transcript(config.transcript_path, config=config, stage_results=stage_results)
        write_json(
            config.trajectory_path,
            build_trajectory_payload(config=config, stage_results=stage_results),
        )
        write_json(
            config.result_path,
            build_result_payload(
                config=config,
                stage_results=stage_results,
                phase1_result=phase1_result,
                phase2_task_results=phase2_task_results,
                started_at=started_at,
                finished_at=utc_now(),
                exception_info=exception_info,
            ),
        )

    logger.info("Run artifacts saved to %s", config.run_dir)
    print(f"Run artifacts saved to: {config.run_dir}")
    print(f"Final notebook saved to: {config.notebook_path}")


def load_config() -> AppConfig:
    api_key = _require_env("OPENROUTER_API_KEY")
    model = _require_env("OPENROUTER_MODEL")
    data_dir = _require_home_credit_data_dir()

    run_id = str(uuid4())
    run_dir = build_run_directory(model)
    notebook_path = run_dir / "notebook.ipynb"

    return AppConfig(
        openrouter_api_key=api_key,
        openrouter_model=model,
        home_credit_data_dir=data_dir,
        notebook_timeout_seconds=NOTEBOOK_TIMEOUT_SECONDS,
        run_id=run_id,
        run_dir=run_dir,
        notebook_path=notebook_path,
        transcript_path=run_dir / "transcript.txt",
        trajectory_path=run_dir / "agent" / "trajectory.json",
        config_path=run_dir / "config.json",
        result_path=run_dir / "result.json",
        log_path=run_dir / "runtime.log",
    )


def bootstrap_notebook(notebook_path: Path) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = new_notebook(cells=[])

    with notebook_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)


def build_phase1_prompt(data_dir: Path) -> str:
    data_dir = data_dir.expanduser().resolve()
    return f"""Starting from a blank notebook, act as an end-to-end data scientist for a given dataset.

Dataset location:
- Home Credit CSV directory: `{data_dir}`

Requirements:
- Add the notebook code needed to import libraries and load the data from the dataset directory above.
- Define `TARGET = 'TARGET'`.
- Use these canonical variable names when a table is loaded: `app_train`, `app_test`, `bureau`, `bureau_bal`, `prev_app`, `pos_cash`, `installments`, `credit_card`.
- For large auxiliary tables, load only the columns needed for each aggregation, downcast dtypes when reasonable, aggregate one table at a time, delete large intermediates, and call `gc.collect()` after heavy steps.
- Do not keep every raw auxiliary table resident in full memory unless it is necessary; phase 2 may reload a missing table from disk when needed.
- Perform a full workflow: data loading, EDA, preprocessing, feature engineering across auxiliary tables, feature selection, deterministic modeling, validation, and interpretation.
- Build reusable customer-level features from the auxiliary tables where useful for the model.
- Train a deterministic binary classifier that supports phase-2 analysis.
- Save the fitted model in `trained_model`.
- Save the selected feature names in `selected_features`.
- Save a validation summary dictionary in `training_metrics`.
- Save any preprocessors, score outputs, engineered datasets, and diagnostics needed for follow-up analysis in the persistent kernel.
- Keep the loaded tables and any derived artifacts available in memory for phase 2.
- Print the key feature counts, validation metrics, and major modeling decisions.
"""


def build_phase2_prompt(data_dir: Path, task: HeadroomTask) -> str:
    data_dir = data_dir.expanduser().resolve()
    return f"""Continue phase 2 headroom analysis in the same persistent notebook kernel.

Task metadata:
- Task ID: `{task.task_id}`
- Task Type: `{task.task_type}`
- Stage: `{task.stage}`
- Difficulty: `{task.difficulty}`
- Failure category: `{task.failure_category}`

Dataset location:
- Home Credit CSV directory: `{data_dir}`

Requirements:
- Reuse the existing notebook state from phase 1, including loaded tables, engineered features, models, and metrics whenever available.
- Add new code cells only as needed.
- Use notebook code execution to inspect state and compute the answer precisely.
- Do not rebuild the entire workflow or retrain the main model unless the question explicitly requires a focused comparison.
- If a required table or object is missing, load or derive only the minimum needed from the dataset directory above.
- Keep useful intermediate results in the notebook so later headroom tasks can reuse them.
- Finish with a concise plain-text answer after executing notebook code.

Question:
{task.prompt}
"""


def build_run_directory(
    model_name: str,
    current_time: datetime | None = None,
) -> Path:
    timestamp = (current_time or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_model_name = sanitize_model_name(model_name)
    return DEFAULT_RUNS_DIR / f"agent_{safe_model_name}_{timestamp}"


def sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
    if sanitized:
        return sanitized
    return "model"


def build_config_payload(config: AppConfig) -> dict[str, object]:
    return {
        "trial_name": config.run_dir.name,
        "trials_dir": str(config.run_dir.parent),
        "job_id": config.run_id,
        "agent": {
            "provider": "openrouter",
            "model_name": config.openrouter_model,
            "phase1_max_steps": PHASE1_MAX_STEPS,
            "phase2_max_steps": PHASE2_MAX_STEPS,
            "notebook_timeout_seconds": config.notebook_timeout_seconds,
            "temperature": AgentConfig.temperature,
        },
        "dataset": {
            "home_credit_data_dir": str(config.home_credit_data_dir),
        },
        "headroom": {
            "source_notebook": "home_credit_headroom_project_new.ipynb",
            "task_count": len(HEADROOM_TASKS),
            "task_types": _count_task_types(HEADROOM_TASKS),
            "tasks": [
                serialize_headroom_task(task, include_source_prompt=True)
                for task in HEADROOM_TASKS
            ],
        },
        "artifacts": {
            "run_dir": str(config.run_dir),
            "notebook_path": str(config.notebook_path),
            "transcript_path": str(config.transcript_path),
            "trajectory_path": str(config.trajectory_path),
            "result_path": str(config.result_path),
            "log_path": str(config.log_path),
        },
        "prompts": {
            "phase1": build_phase1_prompt(config.home_credit_data_dir),
            "phase2_tasks": [
                {
                    **serialize_headroom_task(task),
                    "rendered_prompt": build_phase2_prompt(config.home_credit_data_dir, task),
                }
                for task in HEADROOM_TASKS
            ],
        },
    }


def build_result_payload(
    *,
    config: AppConfig,
    stage_results: list[tuple[str, AgentRunResult]],
    phase1_result: AgentRunResult | None,
    phase2_task_results: list[tuple[HeadroomTask, AgentRunResult]],
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
        "phase1": _serialize_run_summary("phase1", phase1_result),
        "phase2": {
            "n_tasks": len(phase2_task_results),
            "task_types": _count_task_types(task for task, _ in phase2_task_results),
            "results": [
                {
                    **serialize_headroom_task(task),
                    "stage_name": task_stage_name(task),
                    "final_response": result.final_response,
                    "steps_used": result.steps_used,
                    "usage": asdict(result.usage),
                }
                for task, result in phase2_task_results
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
            "notebook": str(config.notebook_path.relative_to(config.run_dir)),
            "transcript": str(config.transcript_path.relative_to(config.run_dir)),
            "trajectory": str(config.trajectory_path.relative_to(config.run_dir)),
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
                "temperature": AgentConfig.temperature,
                "phase1_max_steps": PHASE1_MAX_STEPS,
                "phase2_max_steps": PHASE2_MAX_STEPS,
            },
        },
        "steps": [
            serialize_trace_step(step)
            for step in flatten_trace_steps(stage_results)
        ],
    }


def flatten_trace_steps(stage_results: list[tuple[str, AgentRunResult]]) -> list[AgentTraceStep]:
    return [step for _, result in stage_results for step in result.trace_steps]


def summarize_usage(stage_results: list[tuple[str, AgentRunResult]]) -> AgentUsageSummary:
    prompt_tokens = sum(result.usage.prompt_tokens for _, result in stage_results)
    completion_tokens = sum(result.usage.completion_tokens for _, result in stage_results)
    total_tokens = sum(result.usage.total_tokens for _, result in stage_results)
    cost_usd = sum(result.usage.cost_usd for _, result in stage_results)
    return AgentUsageSummary(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
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
        "observation": {
            "results": [
                {"content": tool_result}
                for tool_result in step.tool_results
            ]
        },
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # Suppress verbose OpenRouter/API request logging from httpx and openai
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def _serialize_run_summary(stage_name: str, result: AgentRunResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "name": stage_name,
        "final_response": result.final_response,
        "steps_used": result.steps_used,
        "usage": asdict(result.usage),
    }


def _count_task_types(tasks: object) -> dict[str, int]:
    counts = {"HT": 0, "HQ": 0}
    for task in tasks:
        if getattr(task, "task_type", None) in counts:
            counts[task.task_type] += 1
    return counts


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if value is None or not value.strip():
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value.strip()


def _require_home_credit_data_dir() -> Path:
    data_dir = Path(_require_env("HOME_CREDIT_DATA_DIR")).expanduser().resolve()
    if not data_dir.exists():
        raise EnvironmentError(f"HOME_CREDIT_DATA_DIR does not exist: {data_dir}")

    required_files = (
        "application_train.csv",
        "application_test.csv",
    )
    missing_files = [file_name for file_name in required_files if not (data_dir / file_name).exists()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise EnvironmentError(
            f"HOME_CREDIT_DATA_DIR is missing required files: {missing} in {data_dir}"
        )

    return data_dir


if __name__ == "__main__":
    main()
