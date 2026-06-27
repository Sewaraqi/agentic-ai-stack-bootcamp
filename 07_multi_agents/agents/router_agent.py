from langchain_core.messages import HumanMessage, SystemMessage

from base.agent_base import AgentBase
from services.agent_registry import AgentRegistry
from services.llm_client import LlmClient


class RouterAgent(AgentBase):
    """
    Concept: Router / Coordinator Pattern

    Receives user input, classifies the intent with ONE LLM call,
    looks up the matching specialist in the registry, and delegates.

    Key design decisions:
        1. The router makes exactly one LLM call (classification only).
           It does NOT run a ReAct loop — that is the specialist's job.
        2. The router uses low temperature for deterministic classification.
           The specialist may use a different temperature for generation.
        3. max_hops prevents oscillation: if routing produces a cycle
           (specialist A routes back to specialist B which routes to A),
           the hop counter trips the circuit and returns an error.
        4. The router does not know which tools each specialist has.
           It only reads role names and descriptions from the registry.
           Adding a new specialist requires no change to RouterAgent.

    Multi-agent failure mode prevented here: oscillation.
        Without max_hops, a misconfigured registry could cause the router
        to keep re-routing indefinitely — each dispatch looks valid in isolation
        but the system never produces an answer.
    """

    def __init__(
        self,
        llm_client: LlmClient,
        registry: AgentRegistry,
        max_hops: int = 3,
    ) -> None:
        self._llm = llm_client
        self._registry = registry
        self._max_hops = max_hops
        self._hop_count = 0
        self._last_route: str = ""

    def chat(self, user_input: str, history: list[dict] | None = None) -> str:
        # Oscillation guard
        if self._hop_count >= self._max_hops:
            self._hop_count = 0
            return (
                f"Routing budget exhausted after {self._max_hops} hops. "
                "Could not find a suitable specialist. Try rephrasing."
            )

        role = self._classify(user_input, history)
        self._last_route = role
        agent = self._registry.get(role)

        if agent is None:
            return (
                f"No specialist found for intent '{role}'. "
                f"Available roles: {self._registry.roles()}"
            )

        self._hop_count += 1
        result = agent.chat(user_input, history)
        self._hop_count = 0
        return result

    def reset(self) -> None:
        self._hop_count = 0
        self._last_route = ""

    def last_route(self) -> str:
        """Returns the role name chosen in the most recent chat() call."""
        return self._last_route

    def _classify(self, user_input: str, history: list[dict] | None = None) -> str:
        """
        Concept: Intent Classification with a Single LLM Call

        Builds a classification prompt from the registry's role descriptions
        and asks the LLM to output exactly one role name.
        history — last few turns are appended so the classifier can route
        vague follow-ups correctly (e.g. "What about there?" → weather_agent).
        """
        descriptions = self._registry.descriptions()
        roles_section = "\n".join(
            f"  {role}: {desc}" for role, desc in descriptions.items()
        )
        history_section = ""
        if history:
            lines = "\n".join(
                f"  {t['role']}: {t['content']}" for t in history
            )
            history_section = f"\n\nConversation history (for context only):\n{lines}"
        system = SystemMessage(content=(
            "You are a routing classifier. "
            "Given a user message, output EXACTLY ONE role name from the list below. "
            "No explanation, no punctuation — just the role name.\n\n"
            f"Available roles:\n{roles_section}"
            f"{history_section}"
        ))
        raw = self._llm.invoke([system, HumanMessage(content=user_input)])
        role = raw.strip().lower().replace(" ", "_")

        # Fallback: substring match if LLM returns a partial name
        if role not in self._registry.roles():
            for r in self._registry.roles():
                if r in role or role in r:
                    return r

        return role