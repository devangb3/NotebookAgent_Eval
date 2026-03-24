from __future__ import annotations

import asyncio
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from harbor import Job
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig, OrchestratorConfig
from harbor.models.trial.config import (
    AgentConfig as HarborAgentConfig,
    ArtifactConfig,
    EnvironmentConfig as HarborEnvironmentConfig,
    TaskConfig as HarborTaskReferenceConfig,
    VerifierConfig as HarborVerifierConfig,
)

from harbor_task_builder import (
    GeneratedHarborTask,
    HarborTaskBuildConfig,
    materialize_harbor_tasks,
)
from task_loader import TaskFile

OPENROUTER_API_BASE = 'https://openrouter.ai/api/v1'


@dataclass(frozen=True)
class RunConfig:
    task_files: tuple[TaskFile, ...]
    agent_name: str | None
    agent_import_path: str | None
    model_name: str | None
    source_name: str
    generated_tasks_dir: Path
    jobs_dir: Path
    job_name: str | None
    n_concurrent_trials: int
    agent_timeout_sec: float
    verifier_timeout_sec: float
    allow_internet: bool
    cpus: int
    memory_mb: int
    storage_mb: int
    force_build: bool
    keep_containers: bool
    debug: bool


def run_harbor_job(config: RunConfig):
    generated_tasks = materialize_harbor_tasks(
        task_files=config.task_files,
        config=HarborTaskBuildConfig(
            output_dir=config.generated_tasks_dir,
            source_name=config.source_name,
            agent_timeout_sec=config.agent_timeout_sec,
            verifier_timeout_sec=config.verifier_timeout_sec,
            allow_internet=config.allow_internet,
            cpus=config.cpus,
            memory_mb=config.memory_mb,
            storage_mb=config.storage_mb,
        ),
    )

    job_config = build_job_config(config=config, generated_tasks=generated_tasks)
    with temporary_env(build_host_env_overrides(config)):
        return asyncio.run(Job(job_config).run())


def build_job_config(
    *,
    config: RunConfig,
    generated_tasks: tuple[GeneratedHarborTask, ...],
) -> JobConfig:
    agent_kwargs = build_agent_kwargs(config)
    agent_env = build_agent_env(config)
    return JobConfig(
        job_name=resolve_job_name(config),
        jobs_dir=config.jobs_dir,
        debug=config.debug,
        orchestrator=OrchestratorConfig(
            n_concurrent_trials=config.n_concurrent_trials,
        ),
        environment=HarborEnvironmentConfig(
            type=EnvironmentType.DOCKER,
            force_build=config.force_build,
            delete=not config.keep_containers,
        ),
        verifier=HarborVerifierConfig(
            override_timeout_sec=config.verifier_timeout_sec,
        ),
        agents=[
            HarborAgentConfig(
                name=config.agent_name,
                import_path=config.agent_import_path,
                model_name=config.model_name,
                override_timeout_sec=config.agent_timeout_sec,
                kwargs=agent_kwargs,
                env=agent_env,
            )
        ],
        tasks=[
            HarborTaskReferenceConfig(path=generated.task_dir)
            for generated in generated_tasks
        ],
        artifacts=[
            ArtifactConfig(
                source='/workspace/final_answer.txt',
                destination='final_answer.txt',
            )
        ],
    )


def resolve_job_name(config: RunConfig) -> str:
    return config.job_name or default_job_name(config)


def default_job_name(config: RunConfig) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    agent = sanitize_name(config.agent_name or 'custom_agent')
    model = sanitize_name(config.model_name or 'no_model')
    return f'dsbench__{agent}__{model}__{timestamp}'


def build_agent_kwargs(config: RunConfig) -> dict[str, str]:
    if not is_openrouter_model(config.model_name):
        return {}

    # Terminus-2 uses LiteLLM in-process, so it needs an explicit API base.
    return {'api_base': get_openrouter_api_base()}


def build_agent_env(config: RunConfig) -> dict[str, str]:
    if not is_openrouter_model(config.model_name):
        return {}

    # Non-secret base URL hints are safe to persist for installed agents.
    api_base = get_openrouter_api_base()
    return {
        'OPENAI_BASE_URL': api_base,
        'OPENAI_API_BASE': api_base,
    }


def build_host_env_overrides(config: RunConfig) -> dict[str, str]:
    if not is_openrouter_model(config.model_name):
        return {}

    openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
    if not openrouter_api_key:
        raise ValueError(
            "Model names prefixed with 'openrouter/' require OPENROUTER_API_KEY in the host environment."
        )

    api_base = get_openrouter_api_base()
    return {
        'OPENROUTER_API_KEY': openrouter_api_key,
        # LiteLLM/OpenAI-compatible clients commonly read OPENAI_API_KEY even
        # when routed through a custom base URL.
        'OPENAI_API_KEY': openrouter_api_key,
        'OPENAI_BASE_URL': api_base,
        'OPENAI_API_BASE': api_base,
    }


def is_openrouter_model(model_name: str | None) -> bool:
    return bool(model_name and model_name.startswith('openrouter/'))


def get_openrouter_api_base() -> str:
    return (
        os.environ.get('OPENROUTER_API_BASE')
        or os.environ.get('OPENROUTER_BASE_URL')
        or OPENROUTER_API_BASE
    )


@contextmanager
def temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    original_values = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def sanitize_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', value).strip('_') or 'value'
