# Added for: main_07_multi_agent.py
from langchain_core.messages import HumanMessage, SystemMessage

from base.agent_base import AgentBase
from agents.router_agent import RouterAgent
from agents.orchestrator_agent import OrchestratorAgent
from services.agent_registry import AgentRegistry
from services.llm_client import LlmClient


class DispatcherAgent(AgentBase):
    """
    Classifies each user query as single-domain or multi-domain with one
    LLM call, then delegates to RouterAgent or OrchestratorAgent.

    single — one specialist can fully answer the query alone
    multi  — the query needs results from more than one specialist role
    """

    def __init__(
        self,
        llm_client: LlmClient,
        registry: AgentRegistry,
        router: RouterAgent,
        orchestrator: OrchestratorAgent,
    ) -> None:
        self._llm = llm_client
        self._registry = registry
        self._router = router
        self._orchestrator = orchestrator
        self._last_mode: str = ""

    def chat(self, user_input: str, history: list[dict] | None = None) -> str:
        mode = self._classify(user_input, history)
        self._last_mode = mode
        if mode == "multi":
            return self._orchestrator.chat(user_input, history)
        return self._router.chat(user_input, history)

    def last_mode(self) -> str:
        """Returns "single" or "multi" from the most recent chat() call."""
        return self._last_mode

    def reset(self) -> None:
        self._last_mode = ""

    def _classify(self, user_input: str, history: list[dict] | None = None) -> str:
        roles_section = "\n".join(
            f"  {role}: {desc}"
            for role, desc in self._registry.descriptions().items()
        )
        history_section = ""
        if history:
            lines = "\n".join(
                f"  {t['role']}: {t['content']}" for t in history
            )
            history_section = f"\n\nConversation history (for context only):\n{lines}"
        system = SystemMessage(content=(
            "You are a query classifier for a multi-agent system.\n\n"
            f"Available specialists:\n{roles_section}\n\n"
            "Rule:\n"
            '  Reply "single" if ONE specialist can fully answer the query alone.\n'
            '  Reply "multi"  if the query needs results from MORE THAN ONE specialist.\n\n'
            "Output only one word — no explanation, no punctuation:  single   or   multi"
            f"{history_section}"
        ))
        raw = self._llm.invoke([system, HumanMessage(content=user_input)])
        result = raw.strip().lower()
        if result in ("single", "multi"):
            return result
        return "single"  # safe fallback for unexpected LLM output