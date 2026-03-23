from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
    return asyncio.run(Job(job_config).run())


def build_job_config(
    *,
    config: RunConfig,
    generated_tasks: tuple[GeneratedHarborTask, ...],
) -> JobConfig:
    return JobConfig(
        job_name=config.job_name or default_job_name(config),
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


def default_job_name(config: RunConfig) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    agent = sanitize_name(config.agent_name or 'custom_agent')
    model = sanitize_name(config.model_name or 'no_model')
    return f'dsbench__{agent}__{model}__{timestamp}'


def sanitize_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', value).strip('_') or 'value'
