from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from run_artifacts import build_run_directory
from task_loader import TaskFile, load_task_files

DEFAULT_DATA_ROOT = Path(".")
NOTEBOOK_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class AppConfig:
    openrouter_api_key: str
    openrouter_model: str
    data_root: Path
    task_files: tuple[TaskFile, ...]
    notebook_timeout_seconds: int
    max_steps: int
    max_workers: int
    run_id: str
    run_dir: Path
    notebook_path: Path
    transcript_path: Path
    trajectory_path: Path
    config_path: Path
    result_path: Path
    log_path: Path
    task_artifacts_dir: Path


def load_config(
    *,
    task_paths: list[str],
    max_steps: int = 20,
    max_workers: int = 4,
) -> AppConfig:
    api_key = _require_env("OPENROUTER_API_KEY")
    model = _require_env("OPENROUTER_MODEL")
    data_root = DEFAULT_DATA_ROOT
    task_files = load_task_files(task_paths, data_root=data_root)

    run_id = str(uuid4())
    run_dir = build_run_directory(model=model)
    notebook_path = run_dir / "notebook.ipynb"

    return AppConfig(
        openrouter_api_key=api_key,
        openrouter_model=model,
        data_root=data_root,
        task_files=task_files,
        notebook_timeout_seconds=NOTEBOOK_TIMEOUT_SECONDS,
        max_steps=max_steps,
        max_workers=max_workers,
        run_id=run_id,
        run_dir=run_dir,
        notebook_path=notebook_path,
        transcript_path=run_dir / "transcript.txt",
        trajectory_path=run_dir / "agent" / "trajectory.json",
        config_path=run_dir / "config.json",
        result_path=run_dir / "result.json",
        log_path=run_dir / "runtime.log",
        task_artifacts_dir=run_dir / "tasks",
    )


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if value is None or not value.strip():
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value.strip()
