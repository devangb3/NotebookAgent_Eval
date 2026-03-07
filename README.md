# DSBench Notebook Agent

Python ReACT agent that edits and executes a persistent Jupyter notebook to run DSBench-style ML tasks and answer follow-up questions in the same kernel session.

## Configuration

Sync the project with `uv`, activate the virtual environment, then copy `.env.example` values into your environment and point `HOME_CREDIT_DATA_DIR` at the extracted Home Credit CSV files.

```bash
uv sync --extra dev
source .venv/bin/activate
cp .env.example .env
```

## Run

Run `main.py` to execute the experiment end-to-end: bootstrap the notebook session, load the Home Credit data, train the model, and run the follow-up headroom question in the same kernel.

Each run saves a notebook in the configured notebook directory using the pattern `agent_{model_name}_{timestamp}.ipynb`.

```bash
source .venv/bin/activate
python main.py
```

## Test

```bash
source .venv/bin/activate
pytest
```
