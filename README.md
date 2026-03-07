# DSBench Notebook Agent

Python ReACT agent that edits and executes a persistent Jupyter notebook to run DSBench-style ML tasks and answer follow-up questions in the same kernel session.

## Configuration

Sync the project with `uv`, activate the virtual environment, then copy `.env.example` values into your environment and point `HOME_CREDIT_DATA_DIR` at the extracted Home Credit CSV files. The run will validate that this directory contains at least `application_train.csv` and `application_test.csv`.

```bash
uv sync --extra dev
source .venv/bin/activate
cp .env.example .env
```

## Run

Run `main.py` to execute the experiment end-to-end: start from a blank notebook, let the agent load the Home Credit data from `HOME_CREDIT_DATA_DIR`, train the model, and run the follow-up headroom question in the same kernel.

Each run creates a Harbor-style artifact folder under `jobs/` using the pattern `agent_{model_name}_{timestamp}`. That folder contains:

- `notebook.ipynb`: the final notebook state
- `transcript.txt`: the full model/tool transcript
- `config.json`: run configuration and prompts
- `result.json`: run metadata, token usage, timings, and total cost
- `agent/trajectory.json`: structured step-by-step trace

```bash
source .venv/bin/activate
python main.py
```

## Test

```bash
source .venv/bin/activate
pytest
```
