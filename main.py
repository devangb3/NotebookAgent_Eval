from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from app_config import load_config
from benchmark_runner import run_benchmark
from run_artifacts import configure_logging
from task_loader import resolve_task_paths

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run notebook benchmark tasks from task JSON files or directories.")
    parser.add_argument(
        "task_paths",
        nargs="+",
        help=("Task paths: a directory (runs all .json files under it), or one or more task JSON files.")
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    resolved_paths = resolve_task_paths(args.task_paths)
    config = load_config(task_paths=[str(p) for p in resolved_paths])
    configure_logging(config.log_path)
    logger.info(
        "Starting run %s | model=%s | tasks=%s",
        config.run_id,
        config.openrouter_model,
        len(config.task_files),
    )
    run_benchmark(config)
    logger.info("Run artifacts saved to %s", config.run_dir)
    print(f"Run artifacts saved to: {config.run_dir}")
    print(f"Final notebook saved to: {config.notebook_path}")


if __name__ == "__main__":
    main()
