# NotebookAgent Eval

NotebookAgent Eval is a reproducible execution harness for evaluating
data-analysis agents inside fresh notebook kernels.

Each task declares its data, question, ground truth, and optional agent
instructions. The runner applies explicit step and concurrency limits, records
the complete model/tool trajectory, and writes enough structured state to
inspect, compare, or reproduce every run.

```text
task JSON → fresh notebook kernel → agent/tool loop
                                      ↓
                         notebook + trajectory
                                      ↓
                     result + usage + failure data
```

## Five-minute quickstart

The project uses `uv` and Python 3.11 or newer.

```bash
uv sync --extra dev
source .venv/bin/activate
cp .env.example .env
```

Set the required values in `.env`:

```dotenv
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=...
```

Run the included task:

```bash
python main.py tasks/home_credit/ht_001.json
```

The run writes its configuration, answer, token usage, timings, notebook,
trajectory, logs, and failure details under `jobs/`.

## Execution model

1. Task definitions are loaded and validated from JSON.
2. Each task receives a new notebook and managed Jupyter kernel.
3. A ReAct-style agent edits and executes notebook cells through typed tools.
4. `--max-steps` bounds each agent loop; `--max-workers` bounds parallel tasks.
5. Successful and failed tasks both retain their notebook and structured
   trajectory.
6. The run-level result aggregates answers, ground truths, step counts, token
   usage, cost, timings, and failure metadata.

Tasks are isolated at the notebook/kernel boundary. If a task exhausts its step
budget or encounters an agent-level error, the remaining tasks continue and the
failed task is recorded with `status`, `failure_type`, and `failure_message`.
An unexpected run-level exception is also written to `exception.txt` and
`result.json`.

## Defining a task

Task data is always resolved under `data/`. Each task JSON declares:

- `task_id`
- `data_source_type`: `table`, `csv`, `images`, or `text`
- `data_source_path`: path relative to `data/`
- `problem_statement`
- `question`
- `ground_truth`
- `agent_instructions`: optional task-specific guidance

Example:

```json
{
  "task_id": "ht_001",
  "data_source_type": "table",
  "data_source_path": "home_credit/application_train.csv",
  "problem_statement": "Analyze the supplied application data.",
  "question": "Which variables are most associated with the target?",
  "ground_truth": "See the task-specific reference answer.",
  "agent_instructions": "Support the answer with notebook evidence."
}
```

See `tasks/home_credit/ht_001.json` for the runnable example.

## Running benchmarks

Task arguments can be individual JSON files or directories containing JSON
tasks.

```bash
# Run every task under a directory
python main.py tasks/

# Run one task
python main.py tasks/home_credit/ht_001.json

# Run a selected set
python main.py tasks/a.json tasks/b.json tasks/c.json
```

Runtime controls:

- `--max-workers N`: maximum tasks executed in parallel; default `4`
- `--max-steps N`: maximum agent steps per task; default `20`

```bash
# More parallelism and a larger per-task budget
python main.py tasks/ --max-workers 8 --max-steps 30

# Sequential execution with one worker
python main.py tasks/ --max-workers 1
```

## Run artifacts

Each invocation creates `jobs/agent_<model>_<timestamp>/`:

```text
jobs/agent_<model>_<timestamp>/
├── config.json
├── result.json
├── transcript.txt
├── runtime.log
├── exception.txt
├── notebook.ipynb
└── tasks/
    └── <task-stage>/
        ├── notebook.ipynb
        └── trajectory.json
```

- `config.json` records the model, limits, task definitions, and artifact paths.
- `result.json` records task statuses, answers, ground truths, step counts,
  token usage, cost, API timings, and run timestamps.
- `transcript.txt` is the human-readable model/tool transcript.
- `runtime.log` contains execution logs.
- `exception.txt` is empty for a normal run and contains traceback information
  for an uncaught run-level failure.
- Each `trajectory.json` stores structured request messages, assistant content,
  tool calls, observations, timestamps, and per-step metrics.
- Each task notebook preserves the final executable analysis state.

A task entry in `result.json` has an explicit outcome:

```json
{
  "task_id": "ht_001",
  "status": "completed",
  "steps_used": 7,
  "failure_type": null,
  "failure_message": null,
  "task_notebook": "tasks/ht_001/notebook.ipynb"
}
```

## Extending the runner

- Add benchmark cases as task JSON plus data under `data/`.
- Add or change notebook tools in `tools.py`.
- Change the agent loop and termination behavior in `agent.py`.
- Change task prompt construction in `prompt_builder.py`.
- Change persisted schemas in `run_artifacts.py`.

Keep new execution paths artifact-complete: a developer should be able to
understand what ran, why it stopped, and what state it left behind.

## Current boundaries

- The runner targets notebook-based data-analysis tasks.
- Model access is currently routed through OpenRouter.
- Ground truth is stored with the run, but benchmark-specific scoring remains
  the responsibility of the surrounding evaluation workflow.
- Kernel isolation is per task; this project is not a container or VM sandbox
  for untrusted code.

## Test

```bash
source .venv/bin/activate
pytest
```
