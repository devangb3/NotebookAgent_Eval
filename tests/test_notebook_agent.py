from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import nbformat
import benchmark_runner
from nbclient.exceptions import DeadKernelError
from nbformat.v4 import new_code_cell, new_notebook

from agent import (
    AgentConfig,
    AgentMaxStepsExceeded,
    AgentProtocolError,
    AgentRunResult,
    AgentStepMetrics,
    AgentTraceStep,
    AgentUsageSummary,
    NotebookReActAgent,
)
from app_config import AppConfig
from benchmark_runner import TaskExecutionRecord, run_benchmark
from environment import NotebookEnvironment, NotebookExecutionFailure
from prompt_builder import build_task_prompt
from run_artifacts import (
    bootstrap_notebook,
    build_config_payload,
    build_result_payload,
    build_run_directory,
    persist_task_notebook,
)
from task_loader import BenchmarkTask, TaskFile, load_task_file, load_task_files, task_stage_name
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
        tools.add_cell(source="print(x * 2)", cell_type="code", position=1)
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


def test_agent_accepts_dict_shaped_chat_completion_responses(tmp_path: Path) -> None:
    notebook_path = tmp_path / "agent-dict-response.ipynb"
    _write_notebook(
        notebook_path,
        [
            new_code_cell("value = 3"),
        ],
    )

    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-add",
                                "function": {
                                    "name": "add_cell",
                                    "arguments": '{"source":"print(value * 4)","cell_type":"code","position":1}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "cost": 0.01,
            },
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-exec",
                                "function": {
                                    "name": "execute_notebook",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
                "cost": 0.02,
            },
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "Completed the task using the persistent notebook kernel.",
                        "tool_calls": [],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 3,
                "total_tokens": 11,
                "cost": 0.03,
            },
        },
    ]

    with NotebookEnvironment(notebook_path) as environment:
        agent = NotebookReActAgent(
            client=FakeClient(responses=responses),
            tools=NotebookToolExecutor(environment),
            config=AgentConfig(model="mock-model", max_steps=5),
        )
        result = agent.run("Multiply the seeded notebook value by four.")

        assert "persistent notebook kernel" in result.final_response
        assert result.usage.total_tokens == 42


def test_agent_prefers_final_answer_tool_for_termination(tmp_path: Path) -> None:
    notebook_path = tmp_path / "agent-final-answer.ipynb"
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
                        content="I have enough evidence.",
                        tool_calls=[
                            FakeToolCall(
                                id="call-final",
                                function=FakeFunctionCall(
                                    name="final_answer",
                                    arguments='{"answer":"12"}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=FakeUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11, cost=0.03),
        ),
    ]

    with NotebookEnvironment(notebook_path) as environment:
        client = FakeClient(responses=responses)
        agent = NotebookReActAgent(
            client=client,
            tools=NotebookToolExecutor(environment),
            config=AgentConfig(model="mock-model", max_steps=5),
        )
        result = agent.run("Multiply the seeded notebook value by four.")

        assert result.final_response == "12"
        assert result.steps_used == 3
        assert len(result.trace_steps) == 3
        assert result.trace_steps[2].tool_calls[0].name == "final_answer"
        assert result.trace_steps[2].tool_results == tuple()
        first_call_tools = client.chat.completions.calls[0]["tools"]
        assert any(
            tool["function"]["name"] == "final_answer"
            for tool in first_call_tools
        )


def test_agent_warns_when_three_steps_from_budget(tmp_path: Path) -> None:
    notebook_path = tmp_path / "agent-warning.ipynb"
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
        client = FakeClient(responses=responses)
        agent = NotebookReActAgent(
            client=client,
            tools=NotebookToolExecutor(environment),
            config=AgentConfig(model="mock-model", max_steps=5),
        )
        result = agent.run("Multiply the seeded notebook value by four.")

        assert "persistent notebook kernel" in result.final_response
        warning_text = "You have only 3 steps remaining before this run is counted as a failure."
        first_request_messages = client.chat.completions.calls[0]["messages"]
        second_request_messages = client.chat.completions.calls[1]["messages"]
        third_request_messages = client.chat.completions.calls[2]["messages"]
        assert all(
            warning_text not in str(message.get("content", ""))
            for message in first_request_messages
        )
        assert all(
            warning_text not in str(message.get("content", ""))
            for message in second_request_messages
        )
        assert any(
            warning_text in str(message.get("content", ""))
            for message in third_request_messages
        )


def test_bootstrap_notebook_starts_empty(tmp_path: Path) -> None:
    notebook_path = tmp_path / "bootstrapped.ipynb"
    bootstrap_notebook(notebook_path=notebook_path)

    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    assert notebook.cells == []


def test_load_task_file_accepts_valid_task(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    source_path = data_root / "dataset.csv"
    source_path.write_text("value\n1\n", encoding="utf-8")
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(_task_payload(task_id="T-001", data_source_path="dataset.csv")), encoding="utf-8")

    task_file = load_task_file(task_path, data_root=data_root)

    assert task_file.task.task_id == "T-001"
    assert task_file.task.resolved_data_source_path(data_root) == source_path.resolve()


def test_load_task_files_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "dataset.csv").write_text("value\n1\n", encoding="utf-8")
    first = tmp_path / "task1.json"
    second = tmp_path / "task2.json"
    payload = _task_payload(task_id="T-001", data_source_path="dataset.csv")
    first.write_text(json.dumps(payload), encoding="utf-8")
    second.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_task_files([str(first), str(second)], data_root=data_root)
    except ValueError as exc:
        assert "Duplicate task_id" in str(exc)
    else:
        raise AssertionError("Expected duplicate task_id validation failure.")


def test_load_task_file_rejects_invalid_data_source_type(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "dataset.csv").write_text("value\n1\n", encoding="utf-8")
    task_path = tmp_path / "task.json"
    payload = _task_payload(task_id="T-001", data_source_path="dataset.csv")
    payload["data_source_type"] = "pdf"
    task_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_task_file(task_path, data_root=data_root)
    except ValueError as exc:
        assert "unsupported data_source_type" in str(exc)
    else:
        raise AssertionError("Expected data source type validation failure.")


def test_load_task_file_rejects_paths_outside_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside_path = tmp_path / "outside.csv"
    outside_path.write_text("value\n1\n", encoding="utf-8")
    task_path = tmp_path / "task.json"
    task_path.write_text(
        json.dumps(_task_payload(task_id="T-001", data_source_path="../outside.csv")),
        encoding="utf-8",
    )

    try:
        load_task_file(task_path, data_root=data_root)
    except ValueError as exc:
        assert "outside the data root" in str(exc)
    else:
        raise AssertionError("Expected data-root validation failure.")


def test_task_stage_name_uses_generic_task_prefix() -> None:
    task = BenchmarkTask(
        task_id="HQ-001",
        data_source_type="csv",
        data_source_path="dataset.csv",
        problem_statement="Context",
        question="Question",
        ground_truth="Answer",
    )

    assert task_stage_name(task) == "task_hq_001"


def test_build_task_prompt_references_source_path_and_contract(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    source_path = data_root / "dataset.csv"
    source_path.write_text("value\n1\n", encoding="utf-8")
    task = BenchmarkTask(
        task_id="T-001",
        data_source_type="csv",
        data_source_path="dataset.csv",
        problem_statement="Understand the dataset.",
        question="What is the average value?",
        ground_truth="1",
        agent_instructions="Keep the answer short.",
    )

    prompt = build_task_prompt(task, data_root=data_root, max_steps=20)

    assert "Starting from a blank notebook" in prompt
    assert "You have at most 20 steps to complete this task." in prompt
    assert "Data Source: csv" in prompt
    assert str(source_path.resolve()) in prompt
    assert "Problem Statement: Understand the dataset." in prompt
    assert "Question: What is the average value?" in prompt
    assert "Task-specific instructions" in prompt
    assert "pandas" in prompt


def test_config_and_result_payloads_include_task_file_metadata(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "dataset.csv").write_text("value\n1\n", encoding="utf-8")
    task = BenchmarkTask(
        task_id="T-001",
        data_source_type="csv",
        data_source_path="dataset.csv",
        problem_statement="Context",
        question="Question?",
        ground_truth="Answer",
        agent_instructions="Short answer.",
    )
    task_file_path = tmp_path / "task.json"
    task_file_path.write_text(json.dumps(_task_payload(task_id="T-001", data_source_path="dataset.csv")), encoding="utf-8")
    task_file = TaskFile(path=task_file_path, task=task)
    config = AppConfig(
        openrouter_api_key="test-key",
        openrouter_model="google/gemini-test",
        data_root=data_root,
        task_files=(task_file,),
        notebook_timeout_seconds=1234,
        max_steps=20,
        max_workers=4,
        run_id="run-123",
        run_dir=tmp_path / "jobs" / "agent_test",
        notebook_path=tmp_path / "jobs" / "agent_test" / "notebook.ipynb",
        transcript_path=tmp_path / "jobs" / "agent_test" / "transcript.txt",
        config_path=tmp_path / "jobs" / "agent_test" / "config.json",
        result_path=tmp_path / "jobs" / "agent_test" / "result.json",
        log_path=tmp_path / "jobs" / "agent_test" / "runtime.log",
        exception_path=tmp_path / "jobs" / "agent_test" / "exception.txt",
        task_artifacts_dir=tmp_path / "jobs" / "agent_test" / "tasks",
    )
    stage_name = task_stage_name(task)
    task_result = AgentRunResult(
        final_response="task answer",
        steps_used=2,
        usage=AgentUsageSummary(prompt_tokens=4, completion_tokens=2, total_tokens=6, cost_usd=0.02),
        trace_steps=tuple(),
    )
    record = TaskExecutionRecord(
        task=task,
        task_file_path=task_file_path,
        stage_name=stage_name,
        result=task_result,
        task_notebook_path=config.task_artifacts_dir / stage_name / "notebook.ipynb",
        task_trajectory_path=config.task_artifacts_dir / stage_name / "trajectory.json",
    )

    config_payload = build_config_payload(config)
    result_payload = build_result_payload(
        config=config,
        stage_results=[(stage_name, task_result)],
        task_results=[record],
        started_at="2026-03-07T00:00:00Z",
        finished_at="2026-03-07T00:01:00Z",
        exception_info=None,
    )

    assert config_payload["tasks"]["count"] == 1
    assert config_payload["tasks"]["task_files"] == [str(task_file_path)]
    assert config_payload["artifacts"]["log_path"].endswith("runtime.log")
    assert result_payload["tasks"]["n_tasks"] == 1
    assert result_payload["tasks"]["n_completed_tasks"] == 1
    assert result_payload["tasks"]["n_failed_tasks"] == 0
    assert result_payload["tasks"]["results"][0]["task_id"] == task.task_id
    assert result_payload["tasks"]["results"][0]["task_file"] == str(task_file_path)
    assert result_payload["tasks"]["results"][0]["ground_truth"] == task.ground_truth
    assert result_payload["tasks"]["results"][0]["status"] == "completed"
    assert result_payload["tasks"]["results"][0]["failure_type"] is None
    assert result_payload["artifacts"]["runtime_log"] == "runtime.log"


def test_run_benchmark_continues_after_max_steps_failure(tmp_path: Path, monkeypatch: object) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "dataset.csv").write_text("value\n1\n", encoding="utf-8")

    task_one = BenchmarkTask(
        task_id="T-001",
        data_source_type="csv",
        data_source_path="dataset.csv",
        problem_statement="Context",
        question="Question one?",
        ground_truth="Answer one",
    )
    task_two = BenchmarkTask(
        task_id="T-002",
        data_source_type="csv",
        data_source_path="dataset.csv",
        problem_statement="Context",
        question="Question two?",
        ground_truth="Answer two",
    )
    task_one_path = tmp_path / "task1.json"
    task_two_path = tmp_path / "task2.json"
    task_one_path.write_text(json.dumps(_task_payload(task_id="T-001", data_source_path="dataset.csv")), encoding="utf-8")
    task_two_path.write_text(json.dumps(_task_payload(task_id="T-002", data_source_path="dataset.csv")), encoding="utf-8")

    config = AppConfig(
        openrouter_api_key="test-key",
        openrouter_model="google/gemini-test",
        data_root=data_root,
        task_files=(
            TaskFile(path=task_one_path, task=task_one),
            TaskFile(path=task_two_path, task=task_two),
        ),
        notebook_timeout_seconds=1234,
        max_steps=20,
        max_workers=2,
        run_id="run-123",
        run_dir=tmp_path / "jobs" / "agent_test",
        notebook_path=tmp_path / "jobs" / "agent_test" / "notebook.ipynb",
        transcript_path=tmp_path / "jobs" / "agent_test" / "transcript.txt",
        config_path=tmp_path / "jobs" / "agent_test" / "config.json",
        result_path=tmp_path / "jobs" / "agent_test" / "result.json",
        log_path=tmp_path / "jobs" / "agent_test" / "runtime.log",
        exception_path=tmp_path / "jobs" / "agent_test" / "exception.txt",
        task_artifacts_dir=tmp_path / "jobs" / "agent_test" / "tasks",
    )

    class _FakeOpenAI:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    def _fake_run_task(
        *,
        client: object,
        config: AppConfig,
        task_file: TaskFile,
        task_index: int,
        total_tasks: int,
    ) -> TaskExecutionRecord:
        del client, total_tasks
        stage_name = task_stage_name(task_file.task)
        task_notebook_path = config.run_dir / f"notebook_{task_index}.ipynb"
        bootstrap_notebook(task_notebook_path)
        if task_index == 1:
            raise AgentMaxStepsExceeded("Agent did not finish within 20 steps.")
        persisted_path = persist_task_notebook(
            source_notebook_path=task_notebook_path,
            task_artifacts_dir=config.task_artifacts_dir,
            stage_name=stage_name,
        )
        result = AgentRunResult(
            final_response="task completed",
            steps_used=1,
            usage=AgentUsageSummary(),
            trace_steps=tuple(),
        )
        return TaskExecutionRecord(
            task=task_file.task,
            task_file_path=task_file.path,
            stage_name=stage_name,
            result=result,
            task_notebook_path=persisted_path,
            task_trajectory_path=config.task_artifacts_dir / stage_name / "trajectory.json",
        )

    monkeypatch.setattr(benchmark_runner, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(benchmark_runner, "_run_task", _fake_run_task)

    run_benchmark(config)

    result_payload = json.loads(config.result_path.read_text(encoding="utf-8"))
    assert result_payload["exception_info"] is None
    assert result_payload["tasks"]["n_tasks"] == 2
    assert result_payload["tasks"]["n_failed_tasks"] == 1
    assert result_payload["tasks"]["n_completed_tasks"] == 1
    by_task_id = {result["task_id"]: result for result in result_payload["tasks"]["results"]}
    assert by_task_id["T-001"]["status"] == "failed"
    assert by_task_id["T-001"]["failure_type"] == "AgentMaxStepsExceeded"
    assert "20 steps" in by_task_id["T-001"]["failure_message"]
    assert by_task_id["T-001"]["final_response"].startswith("FAILED:")
    assert by_task_id["T-002"]["status"] == "completed"
    assert by_task_id["T-002"]["failure_type"] is None


def test_failed_task_persists_partial_trajectory_on_max_steps(tmp_path: Path, monkeypatch: object) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "dataset.csv").write_text("value\n1\n", encoding="utf-8")

    task = BenchmarkTask(
        task_id="T-001",
        data_source_type="csv",
        data_source_path="dataset.csv",
        problem_statement="Context",
        question="Question?",
        ground_truth="Answer",
    )
    task_path = tmp_path / "task1.json"
    task_path.write_text(json.dumps(_task_payload(task_id="T-001", data_source_path="dataset.csv")), encoding="utf-8")

    config = AppConfig(
        openrouter_api_key="test-key",
        openrouter_model="google/gemini-test",
        data_root=data_root,
        task_files=(TaskFile(path=task_path, task=task),),
        notebook_timeout_seconds=1234,
        max_steps=20,
        max_workers=1,
        run_id="run-123",
        run_dir=tmp_path / "jobs" / "agent_test",
        notebook_path=tmp_path / "jobs" / "agent_test" / "notebook.ipynb",
        transcript_path=tmp_path / "jobs" / "agent_test" / "transcript.txt",
        config_path=tmp_path / "jobs" / "agent_test" / "config.json",
        result_path=tmp_path / "jobs" / "agent_test" / "result.json",
        log_path=tmp_path / "jobs" / "agent_test" / "runtime.log",
        exception_path=tmp_path / "jobs" / "agent_test" / "exception.txt",
        task_artifacts_dir=tmp_path / "jobs" / "agent_test" / "tasks",
    )

    class _FakeOpenAI:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    partial_result = AgentRunResult(
        final_response="",
        steps_used=20,
        usage=AgentUsageSummary(prompt_tokens=10, completion_tokens=2, total_tokens=12, cost_usd=0.5),
        trace_steps=(
            AgentTraceStep(
                step_id=1,
                stage="task_T_001",
                timestamp="2026-03-09T00:00:00Z",
                request_messages=({"role": "user", "content": "Task"},),
                assistant_content="working",
                tool_calls=tuple(),
                tool_results=tuple(),
                metrics=AgentStepMetrics(prompt_tokens=10, completion_tokens=2, total_tokens=12, cost_usd=0.5, api_duration_ms=123.0),
            ),
        ),
    )

    def _fake_run_task(
        *,
        client: object,
        config: AppConfig,
        task_file: TaskFile,
        task_index: int,
        total_tasks: int,
    ) -> TaskExecutionRecord:
        del client, total_tasks
        stage_name = task_stage_name(task_file.task)
        task_notebook_path = config.run_dir / f"notebook_{task_index}.ipynb"
        bootstrap_notebook(task_notebook_path)
        raise AgentMaxStepsExceeded("Agent did not finish within 20 steps.", partial_result=partial_result)

    monkeypatch.setattr(benchmark_runner, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(benchmark_runner, "_run_task", _fake_run_task)

    run_benchmark(config)

    result_payload = json.loads(config.result_path.read_text(encoding="utf-8"))
    task_result = result_payload["tasks"]["results"][0]
    assert task_result["status"] == "failed"
    assert task_result["steps_used"] == 20
    assert task_result["usage"]["total_tokens"] == 12

    trajectory_path = config.task_artifacts_dir / task_stage_name(task) / "trajectory.json"
    trajectory_payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert len(trajectory_payload["steps"]) == 1
    assert trajectory_payload["steps"][0]["metrics"]["total_tokens"] == 12


def test_run_benchmark_continues_after_agent_protocol_failure(tmp_path: Path, monkeypatch: object) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "dataset.csv").write_text("value\n1\n", encoding="utf-8")
    first_task = tmp_path / "task1.json"
    second_task = tmp_path / "task2.json"
    first_task.write_text(json.dumps(_task_payload(task_id="T-001", data_source_path="dataset.csv")), encoding="utf-8")
    second_task.write_text(json.dumps(_task_payload(task_id="T-002", data_source_path="dataset.csv")), encoding="utf-8")
    task_files = load_task_files([str(first_task), str(second_task)], data_root=data_root)
    config = AppConfig(
        openrouter_api_key="test-key",
        openrouter_model="openai/gpt-4.1-mini",
        data_root=data_root,
        task_files=task_files,
        notebook_timeout_seconds=1234,
        max_steps=20,
        max_workers=2,
        run_id="run-123",
        run_dir=tmp_path / "jobs" / "agent_test",
        notebook_path=tmp_path / "jobs" / "agent_test" / "notebook.ipynb",
        transcript_path=tmp_path / "jobs" / "agent_test" / "transcript.txt",
        config_path=tmp_path / "jobs" / "agent_test" / "config.json",
        result_path=tmp_path / "jobs" / "agent_test" / "result.json",
        log_path=tmp_path / "jobs" / "agent_test" / "runtime.log",
        exception_path=tmp_path / "jobs" / "agent_test" / "exception.txt",
        task_artifacts_dir=tmp_path / "jobs" / "agent_test" / "tasks",
    )

    class _FakeOpenAI:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    def _fake_run_task(
        *,
        client: object,
        config: AppConfig,
        task_file: TaskFile,
        task_index: int,
        total_tasks: int,
    ) -> TaskExecutionRecord:
        del client, total_tasks
        stage_name = task_stage_name(task_file.task)
        task_notebook_path = config.run_dir / f"notebook_{task_index}.ipynb"
        bootstrap_notebook(task_notebook_path)
        if task_index == 1:
            raise AgentProtocolError("Chat completion response does not contain choices.")
        persisted_path = persist_task_notebook(
            source_notebook_path=task_notebook_path,
            task_artifacts_dir=config.task_artifacts_dir,
            stage_name=stage_name,
        )
        result = AgentRunResult(
            final_response="task completed",
            steps_used=1,
            usage=AgentUsageSummary(),
            trace_steps=tuple(),
        )
        return TaskExecutionRecord(
            task=task_file.task,
            task_file_path=task_file.path,
            stage_name=stage_name,
            result=result,
            task_notebook_path=persisted_path,
            task_trajectory_path=config.task_artifacts_dir / stage_name / "trajectory.json",
        )

    monkeypatch.setattr(benchmark_runner, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(benchmark_runner, "_run_task", _fake_run_task)

    run_benchmark(config)

    result_payload = json.loads(config.result_path.read_text(encoding="utf-8"))
    assert result_payload["exception_info"] is None
    assert result_payload["tasks"]["n_tasks"] == 2
    assert result_payload["tasks"]["n_failed_tasks"] == 1
    assert result_payload["tasks"]["n_completed_tasks"] == 1
    by_task_id = {result["task_id"]: result for result in result_payload["tasks"]["results"]}
    assert by_task_id["T-001"]["status"] == "failed"
    assert by_task_id["T-001"]["failure_type"] == "AgentProtocolError"
    assert "does not contain choices" in by_task_id["T-001"]["failure_message"]
    assert by_task_id["T-002"]["status"] == "completed"
    assert by_task_id["T-002"]["failure_type"] is None


def test_persist_task_notebook_copies_to_task_directory(tmp_path: Path) -> None:
    source_notebook = tmp_path / "notebook.ipynb"
    _write_notebook(source_notebook, [new_code_cell("print('hello')")])

    persisted_path = persist_task_notebook(
        source_notebook_path=source_notebook,
        task_artifacts_dir=tmp_path / "tasks",
        stage_name="task_t_001",
    )

    assert persisted_path == tmp_path / "tasks" / "task_t_001" / "notebook.ipynb"
    assert persisted_path.exists()


def test_build_run_directory_uses_model_name_and_timestamp() -> None:
    run_path = build_run_directory(
        model="openai/gpt-4.1-mini",
        current_time=datetime(2026, 3, 7, 12, 34, 56),
    )

    assert run_path == Path("jobs/agent_openai_gpt_4_1_mini_20260307_123456")


def test_build_run_directory_falls_back_to_model_when_sanitized_name_is_empty() -> None:
    run_path = build_run_directory(
        model="///",
        current_time=datetime(2026, 3, 7, 12, 34, 56),
    )

    assert run_path == Path("jobs/agent_model_20260307_123456")


def _task_payload(*, task_id: str, data_source_path: str) -> dict[str, str]:
    return {
        "task_id": task_id,
        "data_source_type": "csv",
        "data_source_path": data_source_path,
        "problem_statement": "Context",
        "question": "Question?",
        "ground_truth": "Answer",
        "agent_instructions": "Short answer.",
    }


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
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self._call_count = 0
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str,
        temperature: float,
    ) -> object:
        self.calls.append(
            {
                "model": model,
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "tool_choice": tool_choice,
                "temperature": temperature,
            }
        )
        if self._call_count >= len(self._responses):
            raise AssertionError("Mock client received more completion calls than expected.")
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


@dataclass(frozen=True)
class FakeChat:
    completions: FakeCompletions


class FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self._chat = FakeChat(completions=FakeCompletions(responses))

    @property
    def chat(self) -> FakeChat:
        return self._chat
