from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from agent import (
    AgentConfig,
    AgentMaxStepsExceeded,
    AgentRunResult,
    AgentUsageSummary,
    NotebookReActAgent,
)
from app_config import AppConfig
from environment import NotebookEnvironment
from prompt_builder import build_task_prompt
from run_artifacts import (
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
    error_type: str | None = None
    error_message: str | None = None


def run_benchmark(config: AppConfig) -> None:
    client = OpenAI(api_key=config.openrouter_api_key, base_url=OPENROUTER_BASE_URL)
    started_at = utc_now()

    write_json(config.config_path, build_config_payload(config))

    stage_results: list[tuple[str, AgentRunResult]] = []
    task_results: list[TaskExecutionRecord] = []
    exception_info: dict[str, str] | None = None

    try:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(
                    _run_task,
                    client=client,
                    config=config,
                    task_file=task_file,
                    task_index=index,
                    total_tasks=len(config.task_files),
                ): (index, task_file)
                for index, task_file in enumerate(config.task_files, start=1)
            }
            results_by_index: dict[int, TaskExecutionRecord] = {}
            for future in as_completed(futures):
                index, task_file = futures[future]
                stage_name = task_stage_name(task_file.task)
                try:
                    task_result = future.result()
                except AgentMaxStepsExceeded as exc:
                    task_notebook_path = config.run_dir / f"notebook_{index}.ipynb"
                    
                    persisted_path = persist_task_notebook(
                        source_notebook_path=task_notebook_path,
                        task_artifacts_dir=config.task_artifacts_dir,
                        stage_name=stage_name,
                    )
                    
                    failure_result = AgentRunResult(
                        final_response=f"FAILED: {type(exc).__name__}: {exc}",
                        steps_used=config.max_steps,
                        usage=AgentUsageSummary(),
                        trace_steps=tuple(),
                    )
                    results_by_index[index] = TaskExecutionRecord(
                        task=task_file.task,
                        task_file_path=task_file.path,
                        stage_name=stage_name,
                        result=failure_result,
                        task_notebook_path=persisted_path,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    print(
                        f"Task [{index}/{len(config.task_files)}] {task_file.task.task_id} "
                        f"failed: {type(exc).__name__}: {exc}"
                    )
                    continue
                results_by_index[index] = task_result

            for index in sorted(results_by_index):
                task_result = results_by_index[index]
                stage_results.append((task_result.stage_name, task_result.result))
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
        for index in range(1, len(config.task_files) + 1):
            task_notebook = config.run_dir / f"notebook_{index}.ipynb"
            if task_notebook.exists():
                task_notebook.unlink()


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
    task_notebook_path = config.run_dir / f"notebook_{task_index}.ipynb"

    bootstrap_notebook(task_notebook_path)
    with NotebookEnvironment(
        task_notebook_path,
        timeout_seconds=config.notebook_timeout_seconds,
    ) as environment:
        agent = NotebookReActAgent(
            client=client,
            tools=NotebookToolExecutor(environment),
            config=AgentConfig(
                model=config.openrouter_model,
                max_steps=config.max_steps,
            ),
        )
        print(f"Task [{task_index}/{total_tasks}] {task.task_id}")
        result = agent.run(task_prompt, stage_name=stage_name)
        print(result.final_response)
        print()

    persisted_path = persist_task_notebook(
        source_notebook_path=task_notebook_path,
        task_artifacts_dir=config.task_artifacts_dir,
        stage_name=stage_name,
    )
    return TaskExecutionRecord(
        task=task,
        task_file_path=task_file.path,
        stage_name=stage_name,
        result=result,
        task_notebook_path=persisted_path,
    )
