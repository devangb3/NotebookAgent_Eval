from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import nbformat
from dotenv import load_dotenv
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from openai import OpenAI

from agent import AgentConfig, NotebookReActAgent
from environment import NotebookEnvironment
from tools import NotebookToolExecutor

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_NOTEBOOK_PATH = Path("notebooks/home_credit_agent_session.ipynb")

TRAINING_PROMPT = """Using the already loaded Home Credit data, build a deterministic ML workflow in the notebook.

Requirements:
- Work only from data already loaded in memory.
- Use `app_train` and `TARGET`.
- Perform preprocessing suitable for a quick proof of concept.
- Perform deterministic feature selection.
- Train a deterministic binary classifier.
- Save the fitted model in `trained_model`.
- Save the selected feature names in `selected_features`.
- Save a validation summary dictionary in `training_metrics`.
- Print the chosen feature count and validation metrics.
"""

HEADROOM_PROMPT = """Answer a follow-up question in the same persistent kernel without retraining the model.

Requirements:
- Reuse the existing `trained_model`, `selected_features`, and loaded datasets already in memory.
- Add new code cells only as needed.
- Score the first 10 rows of `app_test` using the existing model.
- Report whether the existing model artifact was reused from memory.
- If the model exposes coefficients or feature importances, print the top 5 features.
- Finish with a concise plain-text answer after executing notebook code.
"""


@dataclass(frozen=True)
class AppConfig:
    openrouter_api_key: str
    openrouter_model: str
    notebook_path: Path


def main() -> None:
    load_dotenv()
    config = load_config()

    bootstrap_notebook(notebook_path=config.notebook_path)

    client = OpenAI(api_key=config.openrouter_api_key, base_url=OPENROUTER_BASE_URL)

    with NotebookEnvironment(config.notebook_path) as environment:
        tools = NotebookToolExecutor(environment)
        agent = NotebookReActAgent(client=client, tools=tools, config=AgentConfig(model=config.openrouter_model))

        training_result = agent.run(TRAINING_PROMPT)
        print("Training stage result:")
        print(training_result.final_response)
        print()

        headroom_result = agent.run(HEADROOM_PROMPT)
        print("Headroom QA result:")
        print(headroom_result.final_response)
        print()
        print(f"Notebook saved to: {config.notebook_path}")


def load_config() -> AppConfig:
    api_key = _require_env("OPENROUTER_API_KEY")
    model = _require_env("OPENROUTER_MODEL")

    notebook_path_value = os.getenv("NOTEBOOK_PATH", str(DEFAULT_NOTEBOOK_PATH))
    notebook_path = build_run_notebook_path(Path(notebook_path_value), model)

    return AppConfig(
        openrouter_api_key=api_key,
        openrouter_model=model,
        notebook_path=notebook_path,
    )


def bootstrap_notebook(notebook_path: Path) -> None:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    notebook = new_notebook(
        cells=[
            new_markdown_cell(
                "# Home Credit Agent Session\n\n"
                "This notebook is a deterministic bootstrap for the DSBench notebook "
                "agent."
            ),
            new_code_cell(
                "import os\n"
                "from pathlib import Path\n"
                "\n"
                "import numpy as np\n"
                "import pandas as pd\n"
                "\n"
                "SEED = 42\n"
                "TARGET = 'TARGET'\n"
                "np.random.seed(SEED)\n"
            ),
            new_code_cell(
                "DATA_DIR = Path(os.environ['HOME_CREDIT_DATA_DIR']).expanduser().resolve()\n"
                "TABLE_FILES = {\n"
                "    'app_train': 'application_train.csv',\n"
                "    'app_test': 'application_test.csv',\n"
                "    'bureau': 'bureau.csv',\n"
                "    'bureau_bal': 'bureau_balance.csv',\n"
                "    'prev_app': 'previous_application.csv',\n"
                "    'pos_cash': 'POS_CASH_balance.csv',\n"
                "    'installments': 'installments_payments.csv',\n"
                "    'credit_card': 'credit_card_balance.csv',\n"
                "}\n"
                "\n"
                "def load_home_credit_tables(data_dir: Path) -> dict[str, pd.DataFrame]:\n"
                "    if not data_dir.exists():\n"
                "        raise FileNotFoundError(f'Data directory does not exist: {data_dir}')\n"
                "    tables: dict[str, pd.DataFrame] = {}\n"
                "    for table_name, file_name in TABLE_FILES.items():\n"
                "        table_path = data_dir / file_name\n"
                "        if not table_path.exists():\n"
                "            raise FileNotFoundError(f'Required dataset file missing: {table_path}')\n"
                "        tables[table_name] = pd.read_csv(table_path)\n"
                "    return tables\n"
                "\n"
                "data_tables = load_home_credit_tables(DATA_DIR)\n"
                "app_train = data_tables['app_train'].copy()\n"
                "app_test = data_tables['app_test'].copy()\n"
                "print(f'Loaded Home Credit tables from {DATA_DIR}')\n"
                "print(f'app_train shape: {app_train.shape}')\n"
                "print(f'app_test shape: {app_test.shape}')\n"
                "print(f'TARGET positive rate: {app_train[TARGET].mean():.4f}')\n"
            ),
        ]
    )

    with notebook_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)


def build_run_notebook_path(
    configured_path: Path,
    model_name: str,
    current_time: datetime | None = None,
) -> Path:
    output_dir = configured_path if configured_path.suffix != ".ipynb" else configured_path.parent
    timestamp = (current_time or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_model_name = sanitize_model_name(model_name)
    return output_dir / f"agent_{safe_model_name}_{timestamp}.ipynb"


def sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
    if sanitized:
        return sanitized
    return "model"


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if value is None or not value.strip():
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value.strip()


if __name__ == "__main__":
    main()
