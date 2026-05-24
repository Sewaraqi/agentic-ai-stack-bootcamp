from abc import ABC, abstractmethod


class AgentBase(ABC):
    """
    Contract every agent must satisfy.
    Using ABC enforces this at instantiation time — a class that forgets to
    implement chat() or reset() raises TypeError before any code runs.
    """

    @abstractmethod
    def chat(self, user_input: str) -> str:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    # Context manager support for automatic cleanup (history, connections, sessions)
    def __enter__(self) -> "AgentBase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.reset()
