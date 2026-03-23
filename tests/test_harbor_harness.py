from __future__ import annotations

from pathlib import Path

from harbor.models.environment_type import EnvironmentType

from harbor_runner import RunConfig, build_job_config, default_job_name
from harbor_task_builder import HarborTaskBuildConfig, materialize_harbor_tasks
from task_loader import load_task_files


def _load_sample_task() -> tuple:
    return load_task_files(
        [Path('tasks/task_01/q01.json')],
        data_root=Path('.').resolve(),
    )


def test_materialize_harbor_tasks_creates_expected_structure(tmp_path: Path) -> None:
    task_files = _load_sample_task()
    generated = materialize_harbor_tasks(
        task_files=task_files,
        config=HarborTaskBuildConfig(
            output_dir=tmp_path / 'generated',
            source_name='dsbench',
            agent_timeout_sec=900.0,
            verifier_timeout_sec=120.0,
            allow_internet=True,
            cpus=2,
            memory_mb=4096,
            storage_mb=10240,
        ),
    )

    generated_task = generated[0]
    assert (generated_task.task_dir / 'instruction.md').exists()
    assert (generated_task.task_dir / 'task.toml').exists()
    assert (generated_task.task_dir / 'environment' / 'Dockerfile').exists()
    repo_dockerfile = Path('tasks/task_01/Dockerfile').read_text(encoding='utf-8')
    assert (
        (generated_task.task_dir / 'environment' / 'Dockerfile').read_text(encoding='utf-8')
        == repo_dockerfile
    )
    assert (generated_task.task_dir / 'environment' / 'data' / 'customers.csv').exists()
    assert (generated_task.task_dir / 'tests' / 'test.sh').exists()
    assert (generated_task.task_dir / 'tests' / 'verify_answer.py').exists()
    assert (generated_task.task_dir / 'tests' / 'expected_answer.json').exists()
    assert (generated_task.task_dir / 'solution' / 'solve.sh').exists()


def test_generated_instruction_uses_terminal_workspace_contract(tmp_path: Path) -> None:
    task_files = _load_sample_task()
    generated = materialize_harbor_tasks(
        task_files=task_files,
        config=HarborTaskBuildConfig(
            output_dir=tmp_path / 'generated',
            source_name='dsbench',
            agent_timeout_sec=900.0,
            verifier_timeout_sec=120.0,
            allow_internet=True,
            cpus=2,
            memory_mb=4096,
            storage_mb=10240,
        ),
    )

    instruction = (generated[0].task_dir / 'instruction.md').read_text(encoding='utf-8')
    assert '/workspace/data' in instruction
    assert '/workspace/final_answer.txt' in instruction
    assert 'shell commands' in instruction
    assert 'notebook code' not in instruction


def test_build_job_config_wires_harbor_agent_and_artifacts(tmp_path: Path) -> None:
    task_files = _load_sample_task()
    generated = materialize_harbor_tasks(
        task_files=task_files,
        config=HarborTaskBuildConfig(
            output_dir=tmp_path / 'generated',
            source_name='dsbench',
            agent_timeout_sec=900.0,
            verifier_timeout_sec=120.0,
            allow_internet=True,
            cpus=2,
            memory_mb=4096,
            storage_mb=10240,
        ),
    )
    run_config = RunConfig(
        task_files=task_files,
        agent_name='oracle',
        agent_import_path=None,
        model_name=None,
        source_name='dsbench',
        generated_tasks_dir=tmp_path / 'generated',
        jobs_dir=tmp_path / 'jobs',
        job_name='unit_test_job',
        n_concurrent_trials=1,
        agent_timeout_sec=900.0,
        verifier_timeout_sec=120.0,
        allow_internet=True,
        cpus=2,
        memory_mb=4096,
        storage_mb=10240,
        force_build=False,
        keep_containers=False,
        debug=False,
    )

    job_config = build_job_config(config=run_config, generated_tasks=generated)

    assert job_config.job_name == 'unit_test_job'
    assert job_config.agents[0].name == 'oracle'
    assert job_config.environment.type == EnvironmentType.DOCKER
    assert job_config.artifacts[0].source == '/workspace/final_answer.txt'
    assert job_config.tasks[0].source is None


def test_default_job_name_is_sanitized() -> None:
    run_config = RunConfig(
        task_files=tuple(),
        agent_name='claude-code',
        agent_import_path=None,
        model_name='anthropic/claude-opus-4-1',
        source_name='dsbench',
        generated_tasks_dir=Path('temp/harbor_tasks'),
        jobs_dir=Path('jobs'),
        job_name=None,
        n_concurrent_trials=1,
        agent_timeout_sec=900.0,
        verifier_timeout_sec=120.0,
        allow_internet=True,
        cpus=2,
        memory_mb=4096,
        storage_mb=10240,
        force_build=False,
        keep_containers=False,
        debug=False,
    )

    value = default_job_name(run_config)
    assert value.startswith('dsbench__claude_code__anthropic_claude_opus_4_1__')
