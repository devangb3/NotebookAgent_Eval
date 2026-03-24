from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from harbor.models.task.config import (
    AgentConfig as HarborAgentTimeoutConfig,
    EnvironmentConfig as HarborTaskEnvironmentConfig,
    SolutionConfig as HarborSolutionConfig,
    TaskConfig as HarborTaskFileConfig,
    VerifierConfig as HarborTaskVerifierConfig,
)

from instruction_builder import build_task_instruction
from task_loader import TaskFile, task_stage_name


@dataclass(frozen=True)
class HarborTaskBuildConfig:
    output_dir: Path
    source_name: str
    agent_timeout_sec: float
    verifier_timeout_sec: float
    allow_internet: bool
    cpus: int
    memory_mb: int
    storage_mb: int


@dataclass(frozen=True)
class GeneratedHarborTask:
    task_file: TaskFile
    stage_name: str
    task_dir: Path


def materialize_harbor_tasks(
    *,
    task_files: tuple[TaskFile, ...],
    config: HarborTaskBuildConfig,
) -> tuple[GeneratedHarborTask, ...]:
    output_dir = config.output_dir.expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_tasks: list[GeneratedHarborTask] = []
    for task_file in task_files:
        generated_tasks.append(_materialize_task(task_file=task_file, config=config))

    return tuple(generated_tasks)


def _materialize_task(
    *,
    task_file: TaskFile,
    config: HarborTaskBuildConfig,
) -> GeneratedHarborTask:
    task = task_file.task
    stage_name = task_stage_name(task)
    task_dir = config.output_dir / stage_name
    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    solution_dir = task_dir / "solution"

    environment_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    solution_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "instruction.md").write_text(build_task_instruction(task), encoding="utf-8")
    (task_dir / "task.toml").write_text(
        _build_task_toml(task_file=task_file, config=config),
        encoding="utf-8",
    )

    _copy_data_source(
        source_path=task.resolved_data_source_path(Path(".")),
        destination_dir=environment_dir / "data",
    )

    shutil.copy2(task_file.resolved_dockerfile_path(), environment_dir / "Dockerfile")
    (tests_dir / "test.sh").write_text(_test_sh_text(), encoding="utf-8")
    (tests_dir / "verify_answer.py").write_text(_verify_answer_py_text(), encoding="utf-8")
    (tests_dir / "expected_answer.json").write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "ground_truth": task.ground_truth,
                "question": task.question,
                "original_task_file": str(task_file.path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (solution_dir / "solve.sh").write_text(_solve_sh_text(task.ground_truth), encoding="utf-8")

    (tests_dir / "test.sh").chmod(0o755)
    (solution_dir / "solve.sh").chmod(0o755)

    return GeneratedHarborTask(task_file=task_file, stage_name=stage_name, task_dir=task_dir)


def _copy_data_source(*, source_path: Path, destination_dir: Path) -> None:
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    if source_path.is_dir():
        shutil.copytree(source_path, destination_dir)
        return

    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_dir / source_path.name)


def _build_task_toml(*, task_file: TaskFile, config: HarborTaskBuildConfig) -> str:
    task = task_file.task
    task_config = HarborTaskFileConfig(
        source=config.source_name,
        metadata={
            "task_id": task.task_id,
            "original_task_file": str(task_file.path),
            "original_data_source_path": task.data_source_path,
        },
        agent=HarborAgentTimeoutConfig(timeout_sec=config.agent_timeout_sec),
        verifier=HarborTaskVerifierConfig(timeout_sec=config.verifier_timeout_sec),
        solution=HarborSolutionConfig(),
        environment=HarborTaskEnvironmentConfig(
            build_timeout_sec=900.0,
            cpus=config.cpus,
            memory_mb=config.memory_mb,
            storage_mb=config.storage_mb,
            allow_internet=config.allow_internet,
        ),
    )
    return task_config.model_dump_toml()


def _test_sh_text() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

python3 /tests/verify_answer.py
"""


def _solve_sh_text(ground_truth: str) -> str:
    escaped = ground_truth.replace("\\", "\\\\").replace('"', '\\"')
    return f"""#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace
printf "%s\n" "{escaped}" > /workspace/final_answer.txt
"""


def _verify_answer_py_text() -> str:
    return """from __future__ import annotations

import json
import math
import re
from pathlib import Path

EXPECTED_PATH = Path('/tests/expected_answer.json')
FINAL_ANSWER_TEXT_PATH = Path('/workspace/final_answer.txt')
FINAL_ANSWER_JSON_PATH = Path('/workspace/final_answer.json')
REWARD_PATH = Path('/logs/verifier/reward.txt')
DETAILS_PATH = Path('/logs/verifier/score.json')

MONTH_NAMES = {
    'january': 1,
    'february': 2,
    'march': 3,
    'april': 4,
    'may': 5,
    'june': 6,
    'july': 7,
    'august': 8,
    'september': 9,
    'october': 10,
    'november': 11,
    'december': 12,
}


def main() -> None:
    expected = json.loads(EXPECTED_PATH.read_text(encoding='utf-8'))
    ground_truth = expected['ground_truth']
    predicted = read_prediction()

    score, reason = score_answer(ground_truth, predicted)

    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(f'{score}\\n', encoding='utf-8')
    DETAILS_PATH.write_text(
        json.dumps(
            {
                'task_id': expected['task_id'],
                'ground_truth': ground_truth,
                'predicted': predicted,
                'score': score,
                'reason': reason,
            },
            indent=2,
        )
        + '\\n',
        encoding='utf-8',
    )
    print(reason)


def read_prediction() -> str:
    if FINAL_ANSWER_TEXT_PATH.exists():
        return FINAL_ANSWER_TEXT_PATH.read_text(encoding='utf-8').strip()

    if FINAL_ANSWER_JSON_PATH.exists():
        payload = json.loads(FINAL_ANSWER_JSON_PATH.read_text(encoding='utf-8'))
        if isinstance(payload, dict) and 'answer' in payload:
            return str(payload['answer']).strip()

    return ''


def score_answer(ground_truth: str, predicted: str) -> tuple[float, str]:
    if not predicted:
        return 0.0, 'missing final answer file'

    gt = canonicalize(ground_truth)
    pred = canonicalize(predicted)

    if gt == pred:
        return 1.0, 'exact canonical match'

    if looks_like_date(gt) and gt in pred:
        return 1.0, 'date appears verbatim in prediction'

    if looks_like_year_month(gt) and year_month_matches(gt, pred):
        return 1.0, 'matching reporting month'

    gt_number = parse_single_number(gt)
    if gt_number is not None:
        if numeric_prediction_matches(gt_number, pred):
            return 1.0, 'numeric answer matches after normalization'
        return 0.0, 'numeric answer mismatch'

    if text_label_matches(gt, predicted):
        return 1.0, 'label answer appears in prediction'

    return 0.0, 'prediction does not match ground truth'


def canonicalize(value: str) -> str:
    text = value.strip().strip('`').strip("'").strip('"')
    text = re.sub(r'\\s+', ' ', text)
    return text


def looks_like_date(value: str) -> bool:
    return bool(re.fullmatch(r'\\d{4}-\\d{2}-\\d{2}', value))


def looks_like_year_month(value: str) -> bool:
    return bool(re.fullmatch(r'\\d{4}-\\d{2}', value))


def year_month_matches(ground_truth: str, predicted: str) -> bool:
    if ground_truth in predicted:
        return True

    year, month = ground_truth.split('-')
    month_num = int(month)
    predicted_lower = predicted.lower()
    for name, idx in MONTH_NAMES.items():
        if idx == month_num and name in predicted_lower:
            if year in predicted or not re.search(r'\\b\\d{4}\\b', predicted):
                return True
    return False


def parse_single_number(value: str) -> float | None:
    cleaned = value.replace(',', '').strip()
    if cleaned.endswith('%'):
        cleaned = cleaned[:-1]

    if re.fullmatch(r'[-+]?\\d+(?:\\.\\d+)?', cleaned):
        return float(cleaned)
    return None


def extract_numbers(value: str) -> list[float]:
    matches = re.findall(r'[-+]?\\d+(?:,\\d{3})*(?:\\.\\d+)?', value)
    numbers: list[float] = []
    for match in matches:
        try:
            numbers.append(float(match.replace(',', '')))
        except ValueError:
            continue
    return numbers


def numeric_prediction_matches(ground_truth: float, predicted: str) -> bool:
    direct = parse_single_number(predicted)
    if direct is not None and math.isclose(ground_truth, direct, rel_tol=1e-9, abs_tol=1e-6):
        return True

    for number in extract_numbers(predicted):
        if math.isclose(ground_truth, number, rel_tol=1e-9, abs_tol=1e-6):
            return True

    return False


def text_label_matches(ground_truth: str, predicted_raw: str) -> bool:
    gt = ground_truth.strip()
    if len(gt) <= 2 and gt.upper() == gt:
        pattern = re.compile(rf'(?<![A-Za-z0-9]){re.escape(gt)}(?![A-Za-z0-9])')
        return bool(pattern.search(predicted_raw))

    predicted = canonicalize(predicted_raw).casefold()
    return ground_truth.casefold() == predicted or ground_truth.casefold() in predicted


if __name__ == '__main__':
    main()
"""
