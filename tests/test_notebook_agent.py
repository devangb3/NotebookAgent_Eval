from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient.exceptions import DeadKernelError
from nbformat.v4 import new_code_cell, new_notebook

from agent import AgentConfig, AgentRunResult, AgentUsageSummary, NotebookReActAgent
from environment import NotebookEnvironment, NotebookExecutionFailure
from headroom_tasks import HEADROOM_TASKS, task_stage_name
from main import (
    AppConfig,
    bootstrap_notebook,
    build_config_payload,
    build_phase1_prompt,
    build_phase2_prompt,
    build_result_payload,
    build_run_directory,
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


def test_execute_notebook_translates_dead_kernel_into_execution_failure(tmp_path: Path) -> None:
    notebook_path = tmp_path / "dead-kernel.ipynb"
    _write_notebook(
        notebook_path,
        [
            new_code_cell("print('heavy step')"),
        ],
    )

    with NotebookEnvironment(notebook_path) as environment:
        environment._kernel_started = True

        def _raise_dead_kernel(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise DeadKernelError("Kernel died")

        environment._client.execute_cell = _raise_dead_kernel  # type: ignore[method-assign]

        try:
            environment.execute_notebook()
        except NotebookExecutionFailure as exc:
            assert "kernel died" in str(exc).lower()
            assert "heavy step" in str(exc)
        else:
            raise AssertionError("Expected NotebookExecutionFailure when kernel dies.")


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


def test_headroom_task_registry_contains_expected_questions() -> None:
    assert len(HEADROOM_TASKS) == 20
    assert HEADROOM_TASKS[0].task_id == "HT-001"
    assert HEADROOM_TASKS[9].task_id == "HT-010"
    assert HEADROOM_TASKS[10].task_id == "HQ-001"
    assert HEADROOM_TASKS[-1].task_id == "HQ-010"
    assert task_stage_name(HEADROOM_TASKS[0]) == "phase2_ht_001"


def test_prompt_builders_reference_dataset_path_and_phase_contract(tmp_path: Path) -> None:
    data_dir = tmp_path / "home-credit-default-risk"
    phase1_prompt = build_phase1_prompt(data_dir)
    phase2_prompt = build_phase2_prompt(data_dir, HEADROOM_TASKS[0])

    assert str(data_dir.resolve()) in phase1_prompt
    assert str(data_dir.resolve()) in phase2_prompt
    assert "blank notebook" in phase1_prompt
    assert "app_train" in phase1_prompt
    assert "bureau" in phase1_prompt
    assert "Be memory-conscious" in phase1_prompt
    assert "gc.collect()" in phase1_prompt
    assert "persistent notebook kernel" in phase2_prompt
    assert "Task ID: `HT-001`" in phase2_prompt
    assert "Do not rebuild the entire workflow" in phase2_prompt


def test_config_and_result_payloads_include_phase2_task_metadata(tmp_path: Path) -> None:
    config = AppConfig(
        openrouter_api_key="test-key",
        openrouter_model="openai/test-model",
        home_credit_data_dir=tmp_path / "data",
        run_id="run-123",
        run_dir=tmp_path / "jobs" / "agent_test",
        notebook_path=tmp_path / "jobs" / "agent_test" / "notebook.ipynb",
        transcript_path=tmp_path / "jobs" / "agent_test" / "transcript.txt",
        trajectory_path=tmp_path / "jobs" / "agent_test" / "agent" / "trajectory.json",
        config_path=tmp_path / "jobs" / "agent_test" / "config.json",
        result_path=tmp_path / "jobs" / "agent_test" / "result.json",
        log_path=tmp_path / "jobs" / "agent_test" / "runtime.log",
    )
    phase1_result = AgentRunResult(
        final_response="phase 1 complete",
        steps_used=4,
        usage=AgentUsageSummary(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.1),
        trace_steps=tuple(),
    )
    phase2_result = AgentRunResult(
        final_response="phase 2 answer",
        steps_used=2,
        usage=AgentUsageSummary(prompt_tokens=4, completion_tokens=2, total_tokens=6, cost_usd=0.02),
        trace_steps=tuple(),
    )
    stage_results = [
        ("phase1", phase1_result),
        (task_stage_name(HEADROOM_TASKS[0]), phase2_result),
    ]

    config_payload = build_config_payload(config)
    result_payload = build_result_payload(
        config=config,
        stage_results=stage_results,
        phase1_result=phase1_result,
        phase2_task_results=[(HEADROOM_TASKS[0], phase2_result)],
        started_at="2026-03-07T00:00:00Z",
        finished_at="2026-03-07T00:01:00Z",
        exception_info=None,
    )

    assert config_payload["headroom"]["task_count"] == 20
    assert config_payload["headroom"]["task_types"] == {"HT": 10, "HQ": 10}
    assert config_payload["artifacts"]["log_path"].endswith("runtime.log")
    assert len(config_payload["prompts"]["phase2_tasks"]) == 20
    assert result_payload["phase1"]["name"] == "phase1"
    assert result_payload["phase2"]["n_tasks"] == 1
    assert result_payload["phase2"]["results"][0]["task_id"] == "HT-001"
    assert result_payload["phase2"]["results"][0]["stage_name"] == "phase2_ht_001"
    assert result_payload["phase2"]["task_types"] == {"HT": 1, "HQ": 0}
    assert result_payload["artifacts"]["runtime_log"] == "runtime.log"


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
