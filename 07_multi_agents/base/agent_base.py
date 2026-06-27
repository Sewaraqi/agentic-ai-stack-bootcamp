from abc import ABC, abstractmethod


class AgentBase(ABC):
    """
        Abstract base class for every agent in this tutorial series.

        Concrete agents must implement:
            chat(user_input)  — send a message, get a response
            reset()           — clear all session state

        The context-manager protocol (__enter__ / __exit__) is provided here
        so every agent can be used with `with Agent(...) as agent:` and
        guaranteed to clean up (clear history, close resources) on exit.
        """
    @abstractmethod
    def chat(self, user_input: str) -> str:
        """
        Concept: Agent Interface
        Every agent exposes a single entry point for user input.
        Standardizing this interface means orchestrators, HITL systems,
        and eval harnesses can interact with ANY agent implementation
        without knowing its internals.
        This is Interface Segregation in practice: the caller only
        needs to know about chat() and reset(), nothing else.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Concept: Session Lifecycle Management
        An agent must be able to clear ALL state between sessions:
        conversation history, vector memory, and open resource handles.
        This is also the architectural hook for 'right to be forgotten'
        at the session level — clearing this wipes the agent's knowledge
        of the current user from RAM.
        """
        ...

    def __enter__(self) -> "AgentBase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.reset()
