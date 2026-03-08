from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

VALID_DATA_SOURCE_TYPES = frozenset({"table", "csv", "images", "text"})


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    data_source_type: str
    data_source_path: str
    problem_statement: str
    question: str
    ground_truth: str
    agent_instructions: str = ""

    def resolved_data_source_path(self, data_root: Path) -> Path:
        candidate = Path(self.data_source_path).expanduser()
        if not candidate.is_absolute():
            candidate = data_root / candidate
        return candidate.resolve()


@dataclass(frozen=True)
class TaskFile:
    path: Path
    task: BenchmarkTask


def load_task_file(task_path: Path, *, data_root: Path) -> TaskFile:
    resolved_task_path = task_path.expanduser().resolve()
    if not resolved_task_path.exists():
        raise FileNotFoundError(f"Task file does not exist: {resolved_task_path}")

    payload = json.loads(resolved_task_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Task file must decode to a JSON object: {resolved_task_path}")

    task = _parse_task(payload=payload)
    _validate_task_path(task, data_root=data_root)
    return TaskFile(path=resolved_task_path, task=task)


def load_task_files(task_paths: list[str], *, data_root: Path) -> tuple[TaskFile, ...]:
    if not task_paths:
        raise ValueError("Provide at least one task JSON file path.")

    task_files = tuple(load_task_file(Path(task_path), data_root=data_root) for task_path in task_paths)
    _validate_unique_task_ids(task_files)
    return task_files


def serialize_task(task: BenchmarkTask) -> dict[str, str]:
    return asdict(task)


def task_stage_name(task: BenchmarkTask) -> str:
    safe_task_id = re.sub(r"[^A-Za-z0-9]+", "_", task.task_id.lower()).strip("_")
    return f"task_{safe_task_id or 'task'}"


def count_data_source_types(tasks: list[BenchmarkTask] | tuple[BenchmarkTask, ...]) -> dict[str, int]:
    counts = {data_source_type: 0 for data_source_type in sorted(VALID_DATA_SOURCE_TYPES)}
    for task in tasks:
        if task.data_source_type in counts:
            counts[task.data_source_type] += 1
    return counts


def _parse_task(*, payload: dict[str, object]) -> BenchmarkTask:
    task_id = _require_string(payload, "task_id")
    data_source_type = _require_string(payload, "data_source_type")
    if data_source_type not in VALID_DATA_SOURCE_TYPES:
        valid_values = ", ".join(sorted(VALID_DATA_SOURCE_TYPES))
        raise ValueError(
            f"Task {task_id!r} has unsupported data_source_type {data_source_type!r}. "
            f"Expected one of: {valid_values}."
        )

    return BenchmarkTask(
        task_id=task_id,
        data_source_type=data_source_type,
        data_source_path=_require_string(payload, "data_source_path"),
        problem_statement=_require_string(payload, "problem_statement"),
        question=_require_string(payload, "question"),
        ground_truth=_require_string(payload, "ground_truth"),
        agent_instructions=_optional_string(payload, "agent_instructions"),
    )


def _validate_unique_task_ids(task_files: tuple[TaskFile, ...]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for task_file in task_files:
        task_id = task_file.task.task_id
        if task_id in seen:
            duplicates.append(task_id)
        seen.add(task_id)
    if duplicates:
        raise ValueError(f"Duplicate task_id values are not allowed: {sorted(set(duplicates))}")


def _validate_task_path(task: BenchmarkTask, *, data_root: Path) -> None:
    resolved_data_root = data_root.expanduser().resolve()
    if not resolved_data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {resolved_data_root}")

    resolved_path = task.resolved_data_source_path(resolved_data_root)
    if not _is_relative_to(resolved_path, resolved_data_root):
        raise ValueError(
            f"Task {task.task_id!r} references data outside the data root: {resolved_path}"
        )
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Task {task.task_id!r} references missing data source: {resolved_path}"
        )


def _require_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Task field {key!r} must be a non-empty string.")
    return value.strip()


def _optional_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Task field {key!r} must be a string when provided.")
    return value.strip()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
