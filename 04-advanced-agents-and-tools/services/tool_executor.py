import time
from dataclasses import dataclass
from base.tool_base import ToolBase, ToolResult


@dataclass
class StepTrace:
    step: int
    phase: str           # PLAN | ACT | OBSERVE
    tool_name: str | None
    details: str
    duration: float = 0.0


class ToolExecutor:
    """
    Enforcement layer between the LLM's decisions and the tool implementations.
    Validates tool existence and required args, then runs with exponential-backoff retry.
    Logs every PLAN/ACT/OBSERVE event so you can reconstruct exactly what happened.
    """

    def __init__(self, max_retries: int = 2, base_delay: float = 0.5) -> None:
        self._registry: dict[str, ToolBase] = {}
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._traces: list[StepTrace] = []

    def register(self, tool: ToolBase) -> None:
        self._registry[tool.schema.name] = tool

    def tool_schemas(self) -> list[dict]:
        return [
            {"name": t.schema.name, "description": t.schema.description, "parameters": t.schema.parameters}
            for t in self._registry.values()
        ]

    def execute(self, step: int, tool_name: str, args: dict) -> ToolResult:
        if tool_name not in self._registry:
            self._traces.append(StepTrace(step, "ACT", tool_name, f"FAIL — unknown tool, args={args}"))
            return ToolResult(error=f"Unknown tool '{tool_name}'. Available: {list(self._registry.keys())}")

        tool = self._registry[tool_name]
        missing = [p for p in tool.schema.parameters.get("required", []) if p not in args]
        if missing:
            self._traces.append(StepTrace(step, "ACT", tool_name, f"FAIL — missing args {missing}"))
            return ToolResult(error=f"Missing required arguments: {missing}")

        max_attempts = self._max_retries + 1
        last_result: ToolResult | None = None
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                time.sleep(self._base_delay * (2 ** (attempt - 2)))
            t0 = time.monotonic()
            try:
                last_result = tool.run(**args)
            except Exception as e:
                last_result = ToolResult(error=f"Unhandled exception: {e}")
            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
            label = f"attempt {attempt}/{max_attempts}"
            if last_result.ok:
                self._traces.append(StepTrace(
                    step, "ACT", tool_name,
                    f"OK {label} args={args} → {str(last_result.value)[:80]}",
                    elapsed_ms,
                ))
                return last_result
            self._traces.append(StepTrace(
                step, "ACT", tool_name,
                f"FAIL {label} error: {last_result.error}",
                elapsed_ms,
            ))
            if not last_result.is_idempotent:
                break
        return last_result

    def log_trace(self, step: int, phase: str, tool_name: str | None, details: str) -> None:
        self._traces.append(StepTrace(step, phase, tool_name, details))

    def get_traces(self) -> list[StepTrace]:
        return list(self._traces)

    def clear_traces(self) -> None:
        self._traces.clear()
