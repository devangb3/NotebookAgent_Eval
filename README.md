# DSBench Notebook Agent

Notebook benchmark runner for data-analysis tasks. Each task is defined in its own JSON file, passed on the command line, and executed in a fresh notebook/kernel. The existing top-level artifact layout under `jobs/` is preserved.

## Configuration

Sync the project with `uv`, activate the virtual environment, then copy `.env.example` values into your environment.

```bash
uv sync --extra dev
source .venv/bin/activate
cp .env.example .env
```

Runtime configuration:

- `OPENROUTER_API_KEY`: required
- `OPENROUTER_MODEL`: required

Task data is always resolved under `data/`.

Each task JSON must declare:

- `task_id`
- `data_source_type`: `table`, `csv`, `images`, or `text`
- `data_source_path`: resolved under `data/`
- `problem_statement`
- `question`
- `ground_truth`
- `agent_instructions`: optional

A sample task file is included at `tasks/home_credit/ht_001.json`.

## Run

Pass task paths as positional arguments. Each path can be a directory (runs all `.json` files under it) or a task JSON file.

**Run all tasks in a directory:**
```bash
source .venv/bin/activate
python main.py tasks/
```

**Run a single task file:**
```bash
source .venv/bin/activate
python main.py tasks/home_credit/ht_001.json
```

**Run specific files:**
```bash
source .venv/bin/activate
python main.py tasks/a.json tasks/b.json tasks/c.json
```

**Options:**
- `--max-workers N`: Number of tasks to run in parallel (default: 4)
- `--max-steps N`: Maximum agent steps per task (default: 20)

```bash
# Custom parallelism and steps
python main.py tasks/ --max-workers 8 --max-steps 30

# Sequential (single worker)
python main.py tasks/ --max-workers 1
```

Each run creates a Harbor-style artifact folder under `jobs/` using the pattern `agent_{model_name}_{timestamp}`. The top-level artifact structure is preserved:

- `notebook.ipynb`: final notebook state from the last executed task
- `transcript.txt`: full model/tool transcript across all tasks
- `config.json`: run configuration and resolved task definitions
- `result.json`: run metadata, token usage, timings, task answers, and ground truths
- `runtime.log`: execution logs
- `exception.txt`: exception details if the run failed with an uncaught error, otherwise empty

Per-task artifacts (notebook and trajectory) are stored under `tasks/<task_stage>/` inside the same run directory:
- `tasks/<task_stage>/notebook.ipynb`: notebook state for that task
- `tasks/<task_stage>/trajectory.json`: structured step-by-step trace for that task

## Test

```bash
source .venv/bin/activate
pytest
```
