import time
from base.agent_base import AgentBase
from agents.conversation_agent import ConversationAgent
from services.llm_client import LlmClient


class TimedAgent(AgentBase):
    def __init__(self, llm_client: LlmClient, slow_threshold_s: float = 3.0) -> None:
        self._agent = ConversationAgent(llm_client)
        self._slow_threshold_s = slow_threshold_s
        self._durations: list[float] = []

    def chat(self, user_input: str) -> str:
        start = time.perf_counter()
        response = self._agent.chat(user_input)
        elapsed = time.perf_counter() - start
        self._durations.append(elapsed)
        if elapsed > self._slow_threshold_s:
            print(f"[WARNING: response took {elapsed:.2f}s — threshold is {self._slow_threshold_s:.2f}s]")
        return response

    def reset(self) -> None:
        self._agent.reset()
        self._durations.clear()

    def stats(self) -> dict:
        if not self._durations:
            return {"turns": 0, "avg_s": 0.0, "min_s": 0.0, "max_s": 0.0}
        return {
            "turns": len(self._durations),
            "avg_s": round(sum(self._durations) / len(self._durations), 2),
            "min_s": round(min(self._durations), 2),
            "max_s": round(max(self._durations), 2),
        }
    def history(self) -> str:
        return self._agent.history_text()