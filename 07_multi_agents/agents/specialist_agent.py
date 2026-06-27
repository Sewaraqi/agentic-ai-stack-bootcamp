from dataclasses import dataclass

from agents.tool_agent import ReActConfig, ToolAgent
from services.llm_client import LlmClient
from services.tool_executor import ToolExecutor


@dataclass
class SpecialistConfig:
    """
    Concept: Specialist Identity

    role        — machine-readable name used by RouterAgent and OrchestratorAgent
                  to dispatch tasks. Must be unique in the AgentRegistry.
    description — plain English: what kinds of questions this specialist handles.
                  RouterAgent and OrchestratorAgent embed this description in their
                  classification prompts so the LLM can decide which specialist fits.
                  Ambiguity here is a routing bug — the LLM will pick the wrong agent.
    max_steps   — per-specialist step budget. A weather agent needs 2 steps (call + answer);
                  a multi-tool agent may need more.
    system_hint — optional domain-specific rule injected into the system prompt.
                  e.g. "call query_rewriter before weather when city is not explicit".
    """
    role: str
    description: str
    max_steps: int = 6
    system_hint: str = ""


class SpecialistAgent(ToolAgent):
    """
    Concept: Specialist Agent — Multi-Agent Patterns

    A named ToolAgent with a fixed role and a plain-English description.
    The role and description are the only things RouterAgent and
    OrchestratorAgent know about this agent — they never inspect its tools.

    This is the 'specialist' in the coordinator → specialist → handoff pattern:
        - RouterAgent (coordinator) reads the description to decide when to dispatch here
        - SpecialistAgent executes the task using its own ReAct loop
        - Result is returned to whoever called chat() — the handoff is just a return value

    Extends ToolAgent so it inherits the full Plan → Act → Observe loop.
    Adding role + description is the only change needed to make a ToolAgent
    visible to the multi-agent layer.
    """

    def __init__(
        self,
        llm_client: LlmClient,
        executor: ToolExecutor,
        config: SpecialistConfig,
    ) -> None:
        """
        Concept: Dependency Injection across Agents

        Each specialist receives its own LlmClient and ToolExecutor.
        Two specialists can share an LlmClient (same LLM, different roles)
        but must NOT share a ToolExecutor — step traces and tool registries
        are specialist-specific. Sharing would cause trace entries from one
        specialist to appear in another's log (inconsistent state failure mode).
        """
        super().__init__(
            llm_client, executor,
            ReActConfig(max_steps=config.max_steps, system_hint=config.system_hint),
        )
        self.role = config.role
        self.description = config.description