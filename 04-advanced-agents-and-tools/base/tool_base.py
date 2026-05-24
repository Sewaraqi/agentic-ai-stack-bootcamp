from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolSchema:
    name: str          # machine-readable identifier the LLM writes in its JSON output
    description: str   # plain English — what the LLM reads to decide WHEN to call the tool
    parameters: dict   # JSON Schema object defining every argument


@dataclass
class ToolResult:
    value: Any = None
    error: str = None
    is_idempotent: bool = True  # False means side effects; don't retry

    @property
    def ok(self) -> bool:
        return self.error is None


class ToolBase(ABC):
    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        ...

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        ...
