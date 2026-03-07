from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tools import NotebookToolExecutor, format_notebook_state


SYSTEM_PROMPT = """You are a deterministic autonomous data science agent operating inside a single persistent Jupyter notebook kernel.

Rules:
- Only interact with the notebook through the provided tools.
- Inspect the current notebook state before making changes.
- For code changes, prefer concise code cells with deterministic behavior.
- If notebook execution fails, inspect the traceback, then repair the relevant cell and execute again.
- Do not reload the dataset unless the task explicitly requires it.
- When the task is complete, respond with a short plain-text summary and do not call more tools.
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
class AgentRunResult:
    final_response: str
    steps_used: int


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

    def run(self, prompt: str) -> AgentRunResult:
        initial_state = format_notebook_state(self._tools.environment.get_state())
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

        for step in range(1, self._config.max_steps + 1):
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                tools=self._tools.tool_schemas,
                tool_choice="auto",
                temperature=self._config.temperature,
            )
            message = _extract_response_message(response)
            content = _extract_message_content(message)
            tool_calls = _extract_tool_calls(message)

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

            if not tool_calls:
                if not content:
                    raise AgentProtocolError("Assistant returned neither tool calls nor final text.")
                return AgentRunResult(final_response=content, steps_used=step)

            for tool_call in tool_calls:
                tool_result = self._tools.dispatch(
                    tool_name=tool_call.name,
                    raw_arguments=tool_call.arguments_json,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.tool_call_id,
                        "content": tool_result,
                    }
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
