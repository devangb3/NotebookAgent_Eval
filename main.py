from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from app_config import load_config
from benchmark_runner import run_benchmark
from run_artifacts import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run notebook benchmark tasks from one or more task JSON files.")
    parser.add_argument(
        "task_files",
        nargs="+",
        help="One or more task JSON files. Pass each task as a separate argument.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(task_paths=args.task_files)
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
