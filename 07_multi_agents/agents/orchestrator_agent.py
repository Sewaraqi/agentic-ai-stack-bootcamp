import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from base.agent_base import AgentBase
from services.agent_registry import AgentRegistry
from services.llm_client import LlmClient


class OrchestratorAgent(AgentBase):
    """
    Concept: Hierarchical Agent Architecture

    Handles complex tasks that require more than one specialist by:
        1. DECOMPOSE  — one LLM call breaks the task into subtasks,
                        each assigned to a named specialist role
        2. FAN-OUT    — subtasks are executed sequentially, one specialist per task
        3. SYNTHESIZE — one final LLM call combines all results into a cohesive answer

    Multi-agent failure modes addressed:
        Duplicated work   — embed shared context in each subtask prompt at decompose time
        Inconsistent state — each specialist has its own ToolExecutor and trace log
        Budget exhausted  — max_subtasks caps the fan-out

    Problem left unsolved (next step):
        Fan-out is sequential. Independent subtasks could run in parallel
        (threading / asyncio) to reduce total latency proportionally.
    """

    def __init__(
        self,
        llm_client: LlmClient,
        registry: AgentRegistry,
        max_subtasks: int = 5,
    ) -> None:
        self._llm = llm_client
        self._registry = registry
        self._max_subtasks = max_subtasks
        self._last_subtask_results: list[dict] = []

    def chat(self, user_input: str, history: list[dict] | None = None) -> str:
        # Phase 1: DECOMPOSE
        subtasks = self._decompose(user_input, history)
        if not subtasks:
            return (
                "Could not decompose the task into subtasks. "
                "Try rephrasing with a more specific request."
            )

        # Phase 2: FAN-OUT
        self._last_subtask_results = []
        for subtask in subtasks[: self._max_subtasks]:
            role = subtask.get("role", "")
            task = subtask.get("task", "")
            agent = self._registry.get(role)

            if agent is None:
                result = (
                    f"[ROUTING ERROR] No specialist registered for role '{role}'. "
                    f"Available: {self._registry.roles()}"
                )
            else:
                result = agent.chat(task, history)

            self._last_subtask_results.append(
                {"role": role, "task": task, "result": result}
            )

        # Phase 3:
        return self._synthesize(user_input, self._last_subtask_results)

    def reset(self) -> None:
        self._last_subtask_results = []

    def get_subtask_results(self) -> list[dict]:
        """Returns the per-specialist results from the last chat() call."""
        return list(self._last_subtask_results)

    def _decompose(self, user_input: str, history: list[dict] | None = None) -> list[dict]:
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
            "You are a task decomposer for a multi-agent system.\n"
            "Break the user's request into subtasks. "
            "Each subtask must be handled by exactly one specialist.\n\n"
            f"Available specialists:\n{roles_section}\n\n"
            "Rules:\n"
            "1. Only use role names from the list above — never invent a new role.\n"
            "2. Each subtask must be self-contained — the specialist receives only "
            "   the task string, not the full conversation.\n"
            "3. Output ONLY a JSON array — no prose, no markdown:\n"
            '   [{"role": "role_name", "task": "what to ask the specialist"}]'
            f"{history_section}"
        ))
        raw = self._llm.invoke([system, HumanMessage(content=user_input)])
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().strip("`").strip()
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return []

    def _synthesize(self, original: str, results: list[dict]) -> str:
        results_text = "\n\n".join(
            f"[{r['role']}] Task: {r['task']}\nResult: {r['result']}"
            for r in results
        )
        system = SystemMessage(content=(
            "You are a synthesis agent. "
            "Given the user's original request and the results from specialist agents, "
            "produce a clear, concise, cohesive final answer. "
            "Do not list the raw specialist outputs — integrate them into a single response."
        ))
        prompt = f"Original request: {original}\n\nSpecialist results:\n{results_text}"
        return self._llm.invoke([system, HumanMessage(content=prompt)])