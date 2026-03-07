from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from nbformat.v4 import new_code_cell, new_markdown_cell

from environment import (
    NotebookEnvironment,
    NotebookExecutionFailure,
    NotebookMutationError,
    NotebookState,
)

CellType = Literal["code", "markdown"]


def format_notebook_state(state: NotebookState, *, max_source_chars: int = 600) -> str:
    chunks = [
        f"Notebook path: {state.notebook_path}",
        f"Executed prefix length: {state.executed_prefix_length}",
        f"Dirty index: {state.dirty_index}",
        "Cells:",
    ]

    for cell in state.cells:
        source = _truncate(cell.source.strip(), max_source_chars)
        outputs = _truncate(cell.outputs_summary.strip(), 300)
        chunks.append(
            "\n".join(
                [
                    f"- index: {cell.index}",
                    f"  type: {cell.cell_type}",
                    f"  execution_count: {cell.execution_count}",
                    f"  source:",
                    _indent_block(source or "<empty>"),
                    f"  outputs:",
                    _indent_block(outputs or "<no output>"),
                ]
            )
        )

    return "\n".join(chunks)


@dataclass(frozen=True)
class NotebookToolExecutor:
    environment: NotebookEnvironment

    @property
    def tool_schemas(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_cell",
                    "description": "Return the current state for one notebook cell by index.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "minimum": 0,
                            },
                        },
                        "required": ["index"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_all_cells",
                    "description": "Return the current state for every notebook cell.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "add_cell",
                    "description": "Insert a new code or markdown cell into the notebook.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "Notebook cell source code or markdown.",
                            },
                            "cell_type": {
                                "type": "string",
                                "enum": ["code", "markdown"],
                            },
                            "position": {
                                "type": "integer",
                                "minimum": 0,
                            },
                        },
                        "required": ["source", "cell_type", "position"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "modify_cell",
                    "description": "Replace the source code of an existing notebook cell.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "new_source": {
                                "type": "string",
                                "description": "Replacement source code for the target cell.",
                            },
                        },
                        "required": ["index", "new_source"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_cell",
                    "description": "Delete a notebook cell by index.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "minimum": 0,
                            },
                        },
                        "required": ["index"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_notebook",
                    "description": "Execute pending notebook cells in the persistent kernel.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def get_cell(self, index: int) -> str:
        state = self.environment.get_state()
        if index < 0 or index >= len(state.cells):
            raise NotebookMutationError(f"Cell index out of range: {index} for notebook with {len(state.cells)} cells.")
        cell = state.cells[index]
        source = _indent_block(cell.source.strip() or "<empty>")
        outputs = _indent_block(cell.outputs_summary.strip() or "<no output>")
        return "\n".join(
            [
                f"Cell {index}:",
                f"  type: {cell.cell_type}",
                f"  execution_count: {cell.execution_count}",
                "  source:",
                source,
                "  outputs:",
                outputs,
            ]
        )

    def get_all_cells(self) -> str:
        return format_notebook_state(self.environment.get_state())

    def add_cell(self, source: str, cell_type: CellType, position: int) -> str:
        if cell_type == "code":
            cell = new_code_cell(source=source)
        else:
            cell = new_markdown_cell(source=source)
        self.environment.insert_cell(cell=cell, position=position)
        return self._result_message(
            f"Added {cell_type} cell at index {position}.",
        )

    def modify_cell(self, index: int, new_source: str) -> str:
        current_cell = self.environment.notebook.cells[index]
        cell_type = str(current_cell.cell_type)
        if cell_type == "code":
            replacement = new_code_cell(source=new_source)
        elif cell_type == "markdown":
            replacement = new_markdown_cell(source=new_source)
        else:
            raise NotebookMutationError(f"Unsupported cell type for modification: {cell_type}")

        self.environment.replace_cell(index=index, cell=replacement)
        return self._result_message(f"Modified cell {index}.")

    def delete_cell(self, index: int) -> str:
        removed = self.environment.remove_cell(index=index)
        return self._result_message(f"Deleted {removed.cell_type} cell at index {index}.")

    def execute_notebook(self) -> str:
        try:
            result = self.environment.execute_notebook()
        except NotebookExecutionFailure as exc:
            return self._result_message(
                "Notebook execution failed with traceback:\n" + str(exc)
            )

        return self._result_message(result.output_text)

    def dispatch(self, tool_name: str, raw_arguments: str) -> str:
        arguments = _parse_tool_arguments(raw_arguments)

        if tool_name == "get_cell":
            index = _require_int(arguments, "index")
            return self.get_cell(index=index)

        if tool_name == "get_all_cells":
            if arguments:
                raise NotebookMutationError("get_all_cells does not accept any arguments.")
            return self.get_all_cells()

        if tool_name == "add_cell":
            source = _require_string(arguments, "source")
            cell_type = _require_cell_type(arguments, "cell_type")
            position = _require_int(arguments, "position")
            return self.add_cell(source=source, cell_type=cell_type, position=position)

        if tool_name == "modify_cell":
            index = _require_int(arguments, "index")
            new_source = _require_string(arguments, "new_source")
            return self.modify_cell(index=index, new_source=new_source)

        if tool_name == "delete_cell":
            index = _require_int(arguments, "index")
            return self.delete_cell(index=index)

        if tool_name == "execute_notebook":
            if arguments:
                raise NotebookMutationError("execute_notebook does not accept any arguments.")
            return self.execute_notebook()

        raise NotebookMutationError(f"Unsupported tool call: {tool_name}")

    def _result_message(self, message: str) -> str:
        state = format_notebook_state(self.environment.get_state())
        return f"{message}\n\nCurrent notebook state:\n{state}"


def _parse_tool_arguments(raw_arguments: str) -> dict[str, object]:
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tool arguments are not valid JSON: {raw_arguments}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must decode to a JSON object.")
    return {str(key): value for key, value in parsed.items()}


def _require_string(arguments: dict[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise TypeError(f"Tool argument {key!r} must be a string.")
    return value


def _require_int(arguments: dict[str, object], key: str) -> int:
    value = arguments.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Tool argument {key!r} must be an integer.")
    return value


def _require_cell_type(arguments: dict[str, object], key: str) -> CellType:
    value = _require_string(arguments, key)
    if value not in {"code", "markdown"}:
        raise ValueError(f"Unsupported cell type: {value}")
    return value


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _indent_block(value: str) -> str:
    lines = value.splitlines() or [value]
    return "\n".join(f"    {line}" for line in lines)
