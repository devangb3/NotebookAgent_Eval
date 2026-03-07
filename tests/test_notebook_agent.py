from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_notebook

from agent import AgentConfig, NotebookReActAgent
from environment import NotebookEnvironment
from main import (
    bootstrap_notebook,
    build_headroom_prompt,
    build_run_directory,
    build_training_prompt,
)
from tools import NotebookToolExecutor


def test_notebook_environment_persists_kernel_state(tmp_path: Path) -> None:
    notebook_path = tmp_path / "session.ipynb"
    _write_notebook(
        notebook_path,
        [
            new_code_cell("x = 21\nprint('seeded')"),
        ],
    )

    with NotebookEnvironment(notebook_path) as environment:
        first_result = environment.execute_notebook()
        assert "seeded" in first_result.output_text

        tools = NotebookToolExecutor(environment)
        tools.add_cell(
            source="print(x * 2)",
            cell_type="code",
            position=1,
        )
        second_result = environment.execute_notebook()
        assert "42" in second_result.output_text


def test_notebook_environment_replays_after_earlier_cell_change(tmp_path: Path) -> None:
    notebook_path = tmp_path / "replay.ipynb"
    _write_notebook(
        notebook_path,
        [
            new_code_cell("x = 21"),
            new_code_cell("print(x * 2)"),
        ],
    )

    with NotebookEnvironment(notebook_path) as environment:
        first_result = environment.execute_notebook()
        assert "42" in first_result.output_text

        tools = NotebookToolExecutor(environment)
        tools.modify_cell(index=0, new_source="x = 10")
        second_result = environment.execute_notebook()
        assert "20" in second_result.output_text


def test_execute_notebook_returns_traceback_text(tmp_path: Path) -> None:
    notebook_path = tmp_path / "failure.ipynb"
    _write_notebook(
        notebook_path,
        [
            new_code_cell("raise ValueError('boom')"),
        ],
    )

    with NotebookEnvironment(notebook_path) as environment:
        tools = NotebookToolExecutor(environment)
        result = tools.execute_notebook()
        assert "ValueError" in result
        assert "boom" in result


def test_get_cell_and_get_all_cells_return_current_state(tmp_path: Path) -> None:
    notebook_path = tmp_path / "read_tools.ipynb"
    _write_notebook(
        notebook_path,
        [
            new_code_cell("value = 5\nprint(value)"),
            new_code_cell("print(value * 2)"),
        ],
    )

    with NotebookEnvironment(notebook_path) as environment:
        environment.execute_notebook()
        tools = NotebookToolExecutor(environment)

        single_cell = tools.get_cell(index=1)
        all_cells = tools.get_all_cells()

        assert "Cell 1:" in single_cell
        assert "print(value * 2)" in single_cell
        assert "10" in single_cell
        assert "Cells:" in all_cells
        assert "value = 5" in all_cells
        assert "print(value * 2)" in all_cells


def test_agent_runs_tool_loop_with_mocked_client(tmp_path: Path) -> None:
    notebook_path = tmp_path / "agent.ipynb"
    _write_notebook(
        notebook_path,
        [
            new_code_cell("value = 3"),
        ],
    )

    responses = [
        FakeResponse(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        content="",
                        tool_calls=[
                            FakeToolCall(
                                id="call-add",
                                function=FakeFunctionCall(
                                    name="add_cell",
                                    arguments='{"source":"print(value * 4)","cell_type":"code","position":1}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=FakeUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14, cost=0.01),
        ),
        FakeResponse(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        content="",
                        tool_calls=[
                            FakeToolCall(
                                id="call-exec",
                                function=FakeFunctionCall(
                                    name="execute_notebook",
                                    arguments="{}",
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=FakeUsage(prompt_tokens=12, completion_tokens=5, total_tokens=17, cost=0.02),
        ),
        FakeResponse(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        content="Completed the task using the persistent notebook kernel.",
                        tool_calls=[],
                    )
                )
            ],
            usage=FakeUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11, cost=0.03),
        ),
    ]

    with NotebookEnvironment(notebook_path) as environment:
        agent = NotebookReActAgent(
            client=FakeClient(responses=responses),
            tools=NotebookToolExecutor(environment),
            config=AgentConfig(model="mock-model", max_steps=5),
        )
        result = agent.run("Multiply the seeded notebook value by four.")
        assert "persistent notebook kernel" in result.final_response
        assert result.usage.prompt_tokens == 30
        assert result.usage.completion_tokens == 12
        assert result.usage.total_tokens == 42
        assert result.usage.cost_usd == 0.06
        assert len(result.trace_steps) == 3
        assert result.trace_steps[0].tool_calls[0].name == "add_cell"
        assert "Cell 1 output" in result.trace_steps[1].tool_results[0]
        snapshot = environment.get_state()
        assert snapshot.cells[1].outputs_summary.strip() == "12"


def test_bootstrap_notebook_starts_empty(tmp_path: Path) -> None:
    notebook_path = tmp_path / "bootstrapped.ipynb"
    bootstrap_notebook(notebook_path=notebook_path)

    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    assert notebook.cells == []


def test_prompt_builders_reference_dataset_path_without_in_memory_constraint(tmp_path: Path) -> None:
    data_dir = tmp_path / "home-credit-default-risk"
    training_prompt = build_training_prompt(data_dir)
    headroom_prompt = build_headroom_prompt(data_dir)

    assert str(data_dir.resolve()) in training_prompt
    assert str(data_dir.resolve()) in headroom_prompt
    assert "already loaded" not in training_prompt
    assert "loaded datasets already in memory" not in headroom_prompt
    assert "blank notebook" in training_prompt


def test_build_run_directory_uses_model_name_and_timestamp() -> None:
    run_path = build_run_directory(
        "openai/gpt-4.1-mini",
        current_time=datetime(2026, 3, 7, 12, 34, 56),
    )

    assert run_path == Path("jobs/agent_openai_gpt_4_1_mini_20260307_123456")


def test_build_run_directory_falls_back_to_model_when_sanitized_name_is_empty() -> None:
    run_path = build_run_directory(
        "///",
        current_time=datetime(2026, 3, 7, 12, 34, 56),
    )

    assert run_path == Path("jobs/agent_model_20260307_123456")


def _write_notebook(path: Path, cells: list[object]) -> None:
    notebook = new_notebook(cells=cells)
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)


@dataclass(frozen=True)
class FakeFunctionCall:
    name: str
    arguments: str


@dataclass(frozen=True)
class FakeToolCall:
    id: str
    function: FakeFunctionCall


@dataclass(frozen=True)
class FakeMessage:
    content: str
    tool_calls: list[FakeToolCall]


@dataclass(frozen=True)
class FakeChoice:
    message: FakeMessage


@dataclass(frozen=True)
class FakeResponse:
    choices: list[FakeChoice]
    usage: object | None = None


@dataclass(frozen=True)
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class FakeCompletions:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = responses
        self._call_count = 0

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str,
        temperature: float,
    ) -> FakeResponse:
        del model, messages, tools, tool_choice, temperature
        if self._call_count >= len(self._responses):
            raise AssertionError("Mock client received more completion calls than expected.")
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


@dataclass(frozen=True)
class FakeChat:
    completions: FakeCompletions


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._chat = FakeChat(completions=FakeCompletions(responses))

    @property
    def chat(self) -> FakeChat:
        return self._chat
