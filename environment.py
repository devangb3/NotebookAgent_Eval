from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Awaitable, cast

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbformat import NotebookNode


@dataclass(frozen=True)
class NotebookCellState:
    index: int
    cell_type: str
    source: str
    execution_count: int | None
    outputs_summary: str


@dataclass(frozen=True)
class NotebookState:
    notebook_path: str
    executed_prefix_length: int
    dirty_index: int | None
    cells: tuple[NotebookCellState, ...]


@dataclass(frozen=True)
class NotebookExecutionResult:
    executed_cell_indices: tuple[int, ...]
    output_text: str


class NotebookEnvironmentError(RuntimeError):
    """Base exception for notebook environment failures."""


class NotebookKernelError(NotebookEnvironmentError):
    """Raised when the managed Jupyter kernel is unavailable."""


class NotebookMutationError(NotebookEnvironmentError):
    """Raised when notebook JSON mutations are invalid."""


class NotebookExecutionFailure(NotebookEnvironmentError):
    """Raised when notebook execution fails."""


class NotebookEnvironment:
    """Persistent notebook runtime backed by nbformat and nbclient."""

    def __init__(
        self,
        notebook_path: str | Path,
        *,
        kernel_name: str = "python3",
        timeout_seconds: int = 300,
    ) -> None:
        self.notebook_path = Path(notebook_path)
        if self.notebook_path.suffix != ".ipynb":
            raise ValueError(f"Notebook path must end with .ipynb: {self.notebook_path}")
        if not self.notebook_path.exists():
            raise FileNotFoundError(f"Notebook does not exist: {self.notebook_path}")

        self.kernel_name = kernel_name
        self.timeout_seconds = timeout_seconds
        self._notebook = self._load_notebook()
        self._client = self._create_client()
        self._executed_prefix_length = 0
        self._dirty_index: int | None = 0
        self._next_execution_count = 1
        self._kernel_started = False

    def __enter__(self) -> "NotebookEnvironment":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _load_notebook(self) -> NotebookNode:
        with self.notebook_path.open("r", encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
        if not isinstance(notebook, NotebookNode):
            raise NotebookEnvironmentError("nbformat did not return a NotebookNode.")
        return notebook

    def _create_client(self) -> NotebookClient:
        return NotebookClient(
            self._notebook,
            kernel_name=self.kernel_name,
            timeout=self.timeout_seconds,
            resources={"metadata": {"path": str(self.notebook_path.parent.resolve())}},
        )

    @property
    def notebook(self) -> NotebookNode:
        return self._notebook

    def close(self) -> None:
        if not self._kernel_started:
            return

        kernel_client = self._client.kc
        if kernel_client is None:
            raise NotebookKernelError("Kernel client is missing while closing the environment.")
        kernel_client.stop_channels()

        kernel_manager = self._client.km
        if kernel_manager is None:
            raise NotebookKernelError("Kernel manager is missing while closing the environment.")
        self._run_awaitable(kernel_manager.shutdown_kernel(now=True))

        self._kernel_started = False

    def save(self) -> None:
        with self.notebook_path.open("w", encoding="utf-8") as handle:
            nbformat.write(self._notebook, handle)

    def insert_cell(self, cell: NotebookNode, position: int) -> None:
        if position < 0 or position > len(self._notebook.cells):
            raise NotebookMutationError(
                f"Cell insert position out of range: {position} for notebook with "
                f"{len(self._notebook.cells)} cells."
            )
        self._notebook.cells.insert(position, cell)
        self._mark_dirty(position)
        self.save()

    def replace_cell(self, index: int, cell: NotebookNode) -> None:
        self._validate_cell_index(index)
        self._notebook.cells[index] = cell
        self._mark_dirty(index)
        self.save()

    def remove_cell(self, index: int) -> NotebookNode:
        self._validate_cell_index(index)
        cell = cast(NotebookNode, self._notebook.cells.pop(index))
        self._mark_dirty(index)
        self.save()
        return cell

    def get_state(self) -> NotebookState:
        cells: list[NotebookCellState] = []
        for index, cell in enumerate(self._notebook.cells):
            outputs_summary = ""
            execution_count: int | None = None
            if cell.cell_type == "code":
                execution_count = cast(int | None, cell.get("execution_count"))
                outputs_summary = self._summarize_outputs(cast(list[NotebookNode], cell.get("outputs", [])))
            cells.append(
                NotebookCellState(
                    index=index,
                    cell_type=str(cell.cell_type),
                    source=str(cell.source),
                    execution_count=execution_count,
                    outputs_summary=outputs_summary,
                )
            )
        return NotebookState(
            notebook_path=str(self.notebook_path),
            executed_prefix_length=self._executed_prefix_length,
            dirty_index=self._dirty_index,
            cells=tuple(cells),
        )

    def execute_notebook(self) -> NotebookExecutionResult:
        if self._dirty_index is None:
            return NotebookExecutionResult(
                executed_cell_indices=tuple(),
                output_text="No pending notebook cells to execute.",
            )

        if self._dirty_index < self._executed_prefix_length:
            self._restart_kernel()
            self._clear_outputs(start_index=0)
            start_index = 0
        else:
            self._ensure_kernel()
            self._clear_outputs(start_index=self._dirty_index)
            start_index = self._dirty_index

        executed_indices: list[int] = []
        output_blocks: list[str] = []

        for index in range(start_index, len(self._notebook.cells)):
            cell = cast(NotebookNode, self._notebook.cells[index])
            if cell.cell_type != "code":
                continue

            try:
                executed_cell = self._client.execute_cell(
                    cell,
                    index,
                    execution_count=self._next_execution_count,
                )
            except CellExecutionError as exc:
                self.save()
                self._dirty_index = index
                self._executed_prefix_length = index
                traceback_text = self._extract_error_traceback(cell)
                if not traceback_text:
                    traceback_text = str(exc)
                raise NotebookExecutionFailure(traceback_text) from exc

            execution_count = cast(int | None, executed_cell.get("execution_count"))
            if execution_count is None:
                raise NotebookKernelError(
                    f"nbclient did not assign an execution count to cell {index}."
                )
            self._next_execution_count = execution_count + 1
            executed_indices.append(index)

            cell_output = self._summarize_outputs(cast(list[NotebookNode], executed_cell.get("outputs", [])))
            if cell_output:
                output_blocks.append(f"Cell {index} output:\n{cell_output}")

        self._executed_prefix_length = len(self._notebook.cells)
        self._dirty_index = None
        self.save()

        if output_blocks:
            output_text = "\n\n".join(output_blocks)
        else:
            output_text = "Notebook executed successfully with no captured output."

        return NotebookExecutionResult(
            executed_cell_indices=tuple(executed_indices),
            output_text=output_text,
        )

    def _ensure_kernel(self) -> None:
        if self._kernel_started:
            return

        self._client.km = self._client.create_kernel_manager()
        self._client.start_new_kernel()
        self._client.start_new_kernel_client()

        kernel_client = self._client.kc
        if kernel_client is None:
            raise NotebookKernelError("nbclient did not create a kernel client.")

        self._run_awaitable(kernel_client.wait_for_ready(timeout=self.timeout_seconds))
        self._client.reset_execution_trackers()
        self._kernel_started = True

    def _restart_kernel(self) -> None:
        if self._kernel_started:
            kernel_client = self._client.kc
            if kernel_client is None:
                raise NotebookKernelError("Kernel client missing during kernel restart.")
            kernel_client.stop_channels()

            kernel_manager = self._client.km
            if kernel_manager is None:
                raise NotebookKernelError("Kernel manager missing during kernel restart.")
            self._run_awaitable(kernel_manager.shutdown_kernel(now=True))

        self._client = self._create_client()
        self._kernel_started = False
        self._next_execution_count = 1
        self._ensure_kernel()

    def _clear_outputs(self, *, start_index: int) -> None:
        for index in range(start_index, len(self._notebook.cells)):
            cell = cast(NotebookNode, self._notebook.cells[index])
            if cell.cell_type != "code":
                continue
            cell["outputs"] = []
            cell["execution_count"] = None

    def _mark_dirty(self, index: int) -> None:
        if self._dirty_index is None or index < self._dirty_index:
            self._dirty_index = index

    def _validate_cell_index(self, index: int) -> None:
        if index < 0 or index >= len(self._notebook.cells):
            raise NotebookMutationError(
                f"Cell index out of range: {index} for notebook with {len(self._notebook.cells)} cells."
            )

    def _summarize_outputs(self, outputs: list[NotebookNode]) -> str:
        chunks: list[str] = []
        for output in outputs:
            output_type = str(output.get("output_type", ""))
            if output_type == "stream":
                chunks.append(str(output.get("text", "")).strip())
                continue
            if output_type in {"display_data", "execute_result"}:
                data = output.get("data")
                if isinstance(data, dict):
                    text_value = data.get("text/plain")
                    if isinstance(text_value, str):
                        chunks.append(text_value.strip())
                continue
            if output_type == "error":
                traceback_value = output.get("traceback")
                if isinstance(traceback_value, list):
                    chunks.append("\n".join(str(line) for line in traceback_value).strip())

        filtered_chunks = [chunk for chunk in chunks if chunk]
        return "\n".join(filtered_chunks)

    def _extract_error_traceback(self, cell: NotebookNode) -> str:
        outputs = cast(list[NotebookNode], cell.get("outputs", []))
        for output in outputs:
            if str(output.get("output_type", "")) != "error":
                continue
            traceback_value = output.get("traceback")
            if isinstance(traceback_value, list):
                return "\n".join(str(line) for line in traceback_value)
        return ""

    def _run_awaitable(self, value: object) -> object:
        if inspect.isawaitable(value):
            return asyncio.run(cast(Awaitable[object], value))
        return value
