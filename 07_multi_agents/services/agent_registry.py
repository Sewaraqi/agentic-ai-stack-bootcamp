class AgentRegistry:
    """
    Concept: Agent Registry

    Maps role names to SpecialistAgent instances.

    This is the multi-agent equivalent of ToolExecutor's tool registry.
    RouterAgent and OrchestratorAgent look up agents by role name —
    they never instantiate agents themselves, and they never know which
    tools each specialist carries. The registry is the only coupling point
    between orchestration logic and specialist implementation.

    Separation of concerns:
        AgentRegistry     — who is available and what is their role?
        RouterAgent       — which specialist should handle this input?
        OrchestratorAgent — which specialists are needed for each subtask?
        SpecialistAgent   — how to handle the task using its own tools?
    """

    def __init__(self) -> None:
        self._registry: dict = {}

    def register(self, agent) -> None:
        """
        Concept: Single-Agent vs Multi-Agent Interfaces

        Every specialist is registered by its role name.
        Role names are the message schema between agents — the RouterAgent
        outputs a role name; the registry translates it to a concrete agent.
        Changing which tools a specialist carries requires no change here.
        """
        self._registry[agent.role] = agent

    def get(self, role: str):
        return self._registry.get(role)

    def roles(self) -> list[str]:
        return list(self._registry.keys())

    def descriptions(self) -> dict[str, str]:
        """Returns {role: description} — used by Router and Orchestrator to build prompts."""
        return {role: agent.description for role, agent in self._registry.items()}