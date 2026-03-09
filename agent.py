from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Protocol, cast

from tools import NotebookToolExecutor, format_notebook_state


SYSTEM_PROMPT = """You are a deterministic autonomous data science agent operating inside a single persistent Jupyter notebook kernel.

Rules:
- Only interact with the notebook through the provided tools.
- Inspect the current notebook state before making changes.
- For code changes, prefer concise code cells with deterministic behavior.
- If notebook execution fails, inspect the traceback, then repair the relevant cell and execute again.
- Do not reload the dataset unless the task explicitly requires it.
- Load all task data exclusively from the path provided in the task prompt;
- Once you have enough evidence to answer the task, call the `final_answer` tool with the exact answer.
- Plain-text completion is allowed as a fallback, but `final_answer` is preferred.
"""


@dataclass(frozen=True)
class AgentConfig:
    model: str
    max_steps: int = 20
    temperature: float = 0.0


@dataclass(frozen=True)
class AgentToolCall:
    tool_call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class AgentStepMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    api_duration_ms: float = 0.0


@dataclass(frozen=True)
class AgentTraceStep:
    step_id: int
    stage: str
    timestamp: str
    request_messages: tuple[dict[str, object], ...]
    assistant_content: str
    tool_calls: tuple[AgentToolCall, ...]
    tool_results: tuple[str, ...]
    metrics: AgentStepMetrics


@dataclass(frozen=True)
class AgentUsageSummary:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class AgentRunResult:
    final_response: str
    steps_used: int
    usage: AgentUsageSummary
    trace_steps: tuple[AgentTraceStep, ...]


class AgentError(RuntimeError):
    """Base exception for agent failures."""


class AgentProtocolError(AgentError):
    """Raised when the model returns an invalid response shape."""


class AgentMaxStepsExceeded(AgentError):
    """Raised when the agent does not finish within the step budget."""


class ChatCompletionsApi(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        tool_choice: str,
        temperature: float,
    ) -> object:
        ...


class ChatApi(Protocol):
    completions: ChatCompletionsApi


class OpenAICompatibleClient(Protocol):
    chat: ChatApi


class NotebookReActAgent:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        tools: NotebookToolExecutor,
        *,
        config: AgentConfig,
    ) -> None:
        self._client = client
        self._tools = tools
        self._config = config

    def run(self, prompt: str, *, stage_name: str = "run") -> AgentRunResult:
        initial_state = format_notebook_state(self._tools.environment.get_state())
        tool_schemas = self._tools.tool_schemas
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task:\n{prompt}\n\n"
                    f"Current notebook state:\n{initial_state}"
                ),
            },
        ]
        trace_steps: list[AgentTraceStep] = []
        usage_summary = AgentUsageSummary()

        for step in range(1, self._config.max_steps + 1):
            remaining_steps = self._config.max_steps - step + 1
            if remaining_steps <= 3:
                messages.append(
                    {
                        "role": "user",
                        "content": _build_step_budget_warning(remaining_steps),
                    }
                )
            request_messages = tuple(
                copy.deepcopy(cast(dict[str, object], message)) for message in messages
            )
            started_at = _utc_now()
            started_perf = perf_counter()
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                temperature=self._config.temperature,
            )
            api_duration_ms = (perf_counter() - started_perf) * 1000.0
            message = _extract_response_message(response)
            content = _extract_message_content(message)
            tool_calls = _extract_tool_calls(message)
            step_metrics = _extract_step_metrics(response, api_duration_ms=api_duration_ms)

            assistant_payload: dict[str, object] = {"role": "assistant"}
            if content:
                assistant_payload["content"] = content
            if tool_calls:
                assistant_payload["tool_calls"] = [
                    {
                        "id": call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments_json,
                        },
                    }
                    for call in tool_calls
                ]
            messages.append(assistant_payload)
            tool_results: list[str] = []
            usage_summary = _merge_usage(usage_summary, step_metrics)

            if not tool_calls:
                trace_steps.append(
                    AgentTraceStep(
                        step_id=step,
                        stage=stage_name,
                        timestamp=started_at,
                        request_messages=request_messages,
                        assistant_content=content,
                        tool_calls=tool_calls,
                        tool_results=tuple(),
                        metrics=step_metrics,
                    )
                )
                if not content:
                    raise AgentProtocolError("Assistant returned neither tool calls nor final text.")
                return AgentRunResult(
                    final_response=content,
                    steps_used=step,
                    usage=usage_summary,
                    trace_steps=tuple(trace_steps),
                )

            final_answer_calls = [call for call in tool_calls if call.name == "final_answer"]
            if final_answer_calls:
                final_response = _extract_final_answer(final_answer_calls[0].arguments_json)
                trace_steps.append(
                    AgentTraceStep(
                        step_id=step,
                        stage=stage_name,
                        timestamp=started_at,
                        request_messages=request_messages,
                        assistant_content=content,
                        tool_calls=tool_calls,
                        tool_results=tuple(),
                        metrics=step_metrics,
                    )
                )
                return AgentRunResult(
                    final_response=final_response,
                    steps_used=step,
                    usage=usage_summary,
                    trace_steps=tuple(trace_steps),
                )

            for tool_call in tool_calls:
                tool_result = self._tools.dispatch(
                    tool_name=tool_call.name,
                    raw_arguments=tool_call.arguments_json,
                )
                tool_results.append(tool_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.tool_call_id,
                        "content": tool_result,
                    }
                )
            trace_steps.append(
                AgentTraceStep(
                    step_id=step,
                    stage=stage_name,
                    timestamp=started_at,
                    request_messages=request_messages,
                    assistant_content=content,
                    tool_calls=tool_calls,
                    tool_results=tuple(tool_results),
                    metrics=step_metrics,
                )
            )

        raise AgentMaxStepsExceeded(
            f"Agent did not finish within {self._config.max_steps} steps."
        )


def _extract_response_message(response: object) -> object:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise AgentProtocolError("Chat completion response does not contain choices.")
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise AgentProtocolError("Chat completion choice does not contain a message.")
    return message


def _extract_message_content(message: object) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    raise AgentProtocolError(f"Unsupported assistant content type: {type(content)!r}")


def _extract_tool_calls(message: object) -> tuple[AgentToolCall, ...]:
    raw_tool_calls = getattr(message, "tool_calls", None)
    if raw_tool_calls is None:
        return tuple()
    if not isinstance(raw_tool_calls, list):
        raise AgentProtocolError("Assistant tool_calls payload must be a list.")

    calls: list[AgentToolCall] = []
    for raw_tool_call in raw_tool_calls:
        tool_call_id = getattr(raw_tool_call, "id", None)
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise AgentProtocolError("Tool call is missing a valid id.")

        function_payload = getattr(raw_tool_call, "function", None)
        if function_payload is None:
            raise AgentProtocolError("Tool call is missing its function payload.")

        tool_name = getattr(function_payload, "name", None)
        if not isinstance(tool_name, str) or not tool_name:
            raise AgentProtocolError("Tool call is missing a valid function name.")

        arguments_json = getattr(function_payload, "arguments", None)
        if not isinstance(arguments_json, str):
            raise AgentProtocolError("Tool call arguments must be a JSON string.")

        calls.append(
            AgentToolCall(
                tool_call_id=tool_call_id,
                name=tool_name,
                arguments_json=arguments_json,
            )
        )

    return tuple(calls)


def _extract_final_answer(arguments_json: str) -> str:
    try:
        payload = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise AgentProtocolError("final_answer arguments must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise AgentProtocolError("final_answer arguments must decode to a JSON object.")

    answer = payload.get("answer")
    if not isinstance(answer, str):
        raise AgentProtocolError("final_answer requires a string `answer` field.")

    normalized_answer = answer.strip()
    if not normalized_answer:
        raise AgentProtocolError("final_answer requires a non-empty `answer`.")
    return normalized_answer


def _extract_step_metrics(response: object, *, api_duration_ms: float) -> AgentStepMetrics:
    usage = getattr(response, "usage", None)
    prompt_tokens = _read_int_field(usage, "prompt_tokens")
    completion_tokens = _read_int_field(usage, "completion_tokens")
    total_tokens = _read_int_field(usage, "total_tokens")
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    cost_usd = _read_float_field(usage, "cost")
    if cost_usd == 0.0:
        cost_usd = _read_float_field(usage, "cost_usd")

    return AgentStepMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        api_duration_ms=api_duration_ms,
    )


def _read_int_field(value: object, key: str) -> int:
    raw_value = _read_field(value, key)
    if isinstance(raw_value, int):
        return raw_value
    return 0


def _read_float_field(value: object, key: str) -> float:
    raw_value = _read_field(value, key)
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    return 0.0


def _read_field(value: object, key: str) -> object:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _merge_usage(summary: AgentUsageSummary, metrics: AgentStepMetrics) -> AgentUsageSummary:
    return AgentUsageSummary(
        prompt_tokens=summary.prompt_tokens + metrics.prompt_tokens,
        completion_tokens=summary.completion_tokens + metrics.completion_tokens,
        total_tokens=summary.total_tokens + metrics.total_tokens,
        cost_usd=summary.cost_usd + metrics.cost_usd,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_step_budget_warning(remaining_steps: int) -> str:
    noun = "step" if remaining_steps == 1 else "steps"
    return (
        f"You have only {remaining_steps} {noun} remaining before this run is "
        "counted as a failure. If you have enough evidence, stop now and call "
        "`final_answer`. Do not do redundant checks or extra calculations."
    )
