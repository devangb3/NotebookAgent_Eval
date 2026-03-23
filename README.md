# DSBench Harbor Harness

Terminal-style Harbor harness for DSBench data-science tasks. The benchmark content stays in your existing task JSON files, but execution now happens through Harbor jobs, Harbor task environments, and Harbor-supported agents.

## Configuration

Sync the project with `uv` and activate the virtual environment.

```bash
uv sync --extra dev
source .venv/bin/activate
```

Any model or provider credentials are now determined by the Harbor agent you choose. For example:

- `claude-code` typically needs Anthropic credentials
- `codex` typically needs OpenAI credentials
- `terminus-2` depends on the model backend you point it at
- `oracle` needs no model credentials and is useful for smoke tests

Each task JSON must declare:

- `task_id`
- `data_source_path`: resolved under the repo root
- `problem_statement`
- `question`
- `ground_truth`
- `agent_instructions`: optional

The same directory as the JSON must contain a `Dockerfile` (one per task family, e.g. `tasks/task_05/Dockerfile` for all `tasks/task_05/q*.json`). The Harbor builder copies it into each generated `environment/Dockerfile`.

At runtime, the harness converts those task JSONs into temporary Harbor task directories under `temp/harbor_tasks/`. Each generated Harbor task contains:

- `instruction.md`
- `task.toml`
- `environment/Dockerfile`
- `tests/test.sh`
- `tests/verify_answer.py`
- `solution/solve.sh` for Harbor’s `oracle` agent

## Run

Pass task paths as positional arguments. Each path can be a directory (runs all `.json` files under it) or a task JSON file.

Run all tasks with a Harbor agent:

```bash
source .venv/bin/activate
python main.py tasks/ --agent terminus-2 --model openai/gpt-5
```

Run a single task with Harbor’s Oracle agent for a local smoke test:

```bash
source .venv/bin/activate
python main.py tasks/task_01/q01.json --agent oracle --job-name smoke_oracle
```

Run with an installed Harbor agent such as Claude Code:

```bash
python main.py tasks/ --agent claude-code --model anthropic/claude-opus-4-1
```

Useful options:

- `--agent-import-path module.path:ClassName`: use a custom Harbor agent
- `--n-concurrent N`: Harbor trial concurrency
- `--agent-timeout-sec S`: per-task agent timeout
- `--verifier-timeout-sec S`: per-task verifier timeout
- `--allow-internet/--no-allow-internet`: task-environment internet policy
- `--jobs-dir DIR`: Harbor jobs output directory
- `--generated-tasks-dir DIR`: scratch directory for generated Harbor tasks

Each run creates a Harbor job under `jobs/` with Harbor-native outputs, for example:

```text
jobs/<job-name>/
├── config.json
├── job.log
├── result.json
└── <trial-name>/
    ├── agent/
    │   └── trajectory.json
    ├── verifier/
    │   ├── reward.txt
    │   └── score.json
    ├── artifacts/
    │   └── final_answer.txt
    ├── config.json
    ├── result.json
    └── trial.log
```

This is Harbor’s native job/trial layout, not the previous notebook-oriented artifact schema.

## Test

```bash
source .venv/bin/activate
pytest
```
