from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from harbor_runner import RunConfig, resolve_job_name, run_harbor_job
from task_loader import load_task_files, resolve_task_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run DSBench tasks through Harbor using local Harbor task directories and Harbor agents.',
    )
    parser.add_argument(
        'task_paths',
        nargs='+',
        help='Task paths: a directory (runs all .json files under it), or one or more task JSON files.',
    )
    parser.add_argument(
        '--agent',
        default='terminus-2',
        help='Harbor agent name, for example terminus-2, claude-code, codex, or oracle.',
    )
    parser.add_argument(
        '--agent-import-path',
        default=None,
        help='Optional custom Harbor agent import path in module:Class form.',
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Optional model name passed through to the Harbor agent.',
    )
    parser.add_argument(
        '--job-name',
        default=None,
        help='Optional Harbor job name. Defaults to a generated dsbench__<agent>__<model>__timestamp value.',
    )
    parser.add_argument(
        '--jobs-dir',
        default='jobs',
        help='Directory where Harbor job outputs will be written.',
    )
    parser.add_argument(
        '--generated-tasks-dir',
        default='temp/harbor_tasks',
        help='Scratch directory for generated Harbor task definitions.',
    )
    parser.add_argument(
        '--source-name',
        default='dsbench',
        help='Source label attached to Harbor task configs and job results.',
    )
    parser.add_argument(
        '--n-concurrent',
        type=int,
        default=4,
        help='Number of Harbor trials to run in parallel.',
    )
    parser.add_argument(
        '--agent-timeout-sec',
        type=float,
        default=900.0,
        help='Maximum Harbor agent execution time per task in seconds.',
    )
    parser.add_argument(
        '--verifier-timeout-sec',
        type=float,
        default=120.0,
        help='Maximum Harbor verifier time per task in seconds.',
    )
    parser.add_argument(
        '--allow-internet',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Whether generated Harbor task environments allow internet access.',
    )
    parser.add_argument('--cpus', type=int, default=2, help='CPU allocation per Harbor task environment.')
    parser.add_argument('--memory-mb', type=int, default=4096, help='Memory allocation per Harbor task environment in MB.')
    parser.add_argument('--storage-mb', type=int, default=10240, help='Storage allocation per Harbor task environment in MB.')
    parser.add_argument(
        '--force-build',
        action='store_true',
        help='Force Harbor to rebuild task environments instead of reusing cached images.',
    )
    parser.add_argument(
        '--keep-containers',
        action='store_true',
        help='Keep Harbor containers around after a run for debugging.',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable Harbor debug logging.',
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    resolved_paths = resolve_task_paths(args.task_paths)
    task_files = load_task_files(resolved_paths, data_root=Path('.').resolve())

    run_config = RunConfig(
        task_files=task_files,
        agent_name=None if args.agent_import_path else args.agent,
        agent_import_path=args.agent_import_path,
        model_name=args.model,
        source_name=args.source_name,
        generated_tasks_dir=Path(args.generated_tasks_dir),
        jobs_dir=Path(args.jobs_dir),
        job_name=args.job_name,
        n_concurrent_trials=args.n_concurrent,
        agent_timeout_sec=args.agent_timeout_sec,
        verifier_timeout_sec=args.verifier_timeout_sec,
        allow_internet=args.allow_internet,
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        storage_mb=args.storage_mb,
        force_build=args.force_build,
        keep_containers=args.keep_containers,
        debug=args.debug,
    )
    resolved_job_name = resolve_job_name(run_config)
    run_config = replace(run_config, job_name=resolved_job_name)
    run_harbor_job(run_config)
    job_dir = Path(args.jobs_dir) / resolved_job_name
    print(f'Harbor job complete: {job_dir}')


if __name__ == '__main__':
    main()
