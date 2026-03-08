from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from agent import AgentConfig, AgentRunResult, NotebookReActAgent
from app_config import AppConfig
from environment import NotebookEnvironment
from prompt_builder import build_task_prompt
from run_artifacts import (
    TASK_MAX_STEPS,
    bootstrap_notebook,
    build_config_payload,
    build_result_payload,
    build_trajectory_payload,
    persist_task_notebook,
    utc_now,
    write_json,
    write_transcript,
)
from task_loader import BenchmarkTask, TaskFile, task_stage_name
from tools import NotebookToolExecutor

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class TaskExecutionRecord:
    task: BenchmarkTask
    task_file_path: Path
    stage_name: str
    result: AgentRunResult
    task_notebook_path: Path


def run_benchmark(config: AppConfig) -> None:
    client = OpenAI(api_key=config.openrouter_api_key, base_url=OPENROUTER_BASE_URL)
    started_at = utc_now()

    write_json(config.config_path, build_config_payload(config))
    bootstrap_notebook(config.notebook_path)

    stage_results: list[tuple[str, AgentRunResult]] = []
    task_results: list[TaskExecutionRecord] = []
    exception_info: dict[str, str] | None = None

    try:
        for index, task_file in enumerate(config.task_files, start=1):
            task_result = _run_task(
                client=client,
                config=config,
                task_file=task_file,
                task_index=index,
                total_tasks=len(config.task_files),
            )
            stage_name = task_stage_name(task_file.task)
            stage_results.append((stage_name, task_result.result))
            task_results.append(task_result)
    except Exception as exc:
        exception_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
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
                task_results=task_results,
                started_at=started_at,
                finished_at=utc_now(),
                exception_info=exception_info,
            ),
        )


def _run_task(
    *,
    client: OpenAI,
    config: AppConfig,
    task_file: TaskFile,
    task_index: int,
    total_tasks: int,
) -> TaskExecutionRecord:
    task = task_file.task
    stage_name = task_stage_name(task)
    task_prompt = build_task_prompt(task, data_root=config.data_root)

    bootstrap_notebook(config.notebook_path)
    with NotebookEnvironment(
        config.notebook_path,
        timeout_seconds=config.notebook_timeout_seconds,
    ) as environment:
        agent = NotebookReActAgent(
            client=client,
            tools=NotebookToolExecutor(environment),
            config=AgentConfig(model=config.openrouter_model, max_steps=TASK_MAX_STEPS),
        )
        print(f"Task [{task_index}/{total_tasks}] {task.task_id}")
        result = agent.run(task_prompt, stage_name=stage_name)
        print(result.final_response)
        print()

    task_notebook_path = persist_task_notebook(
        source_notebook_path=config.notebook_path,
        task_artifacts_dir=config.task_artifacts_dir,
        stage_name=stage_name,
    )
    return TaskExecutionRecord(
        task=task,
        task_file_path=task_file.path,
        stage_name=stage_name,
        result=result,
        task_notebook_path=task_notebook_path,
    )
