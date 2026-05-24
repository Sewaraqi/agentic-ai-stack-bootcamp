import json
import re
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic.dataclasses import dataclass as pydantic_dataclass

from base.agent_base import AgentBase
from services.tool_executor import ToolExecutor
from services.llm_client import LlmClient


@dataclass
class ReActConfig:
    max_steps: int = 6       # hard cap on Plan/Act/Observe iterations
    max_answer_length: int = 600


def _parse_json(text: str) -> dict | None:
    """
    Extract a JSON object from the LLM's raw response.
    Handles the common case where the LLM wraps JSON in markdown code fences
    even when explicitly told not to — a known LLM behaviour.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().strip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _build_system_prompt(tool_schemas: list[dict]) -> str:
    tools_section = ""
    for tool in tool_schemas:
        props = tool["parameters"].get("properties", {})
        required_args = tool["parameters"].get("required", [])
        args_lines = ""
        for pname, pdef in props.items():
            req = " (required)" if pname in required_args else " (optional)"
            enum_hint = (
                f" — one of: {', '.join(str(v) for v in pdef['enum'])}" if "enum" in pdef else ""
            )
            args_lines += f"\n    {pname} ({pdef['type']}{req}{enum_hint}): {pdef.get('description', '')}"
        tools_section += (
            f"\nTool: {tool['name']}\n"
            f"Description: {tool['description']}\n"
            f"Arguments: {args_lines}\n"
        )
    return f"""You are a helpful assistant with access to tools.

RESPONSE FORMAT — follow exactly:
- To call a tool, output ONLY this JSON (one object, no surrounding text):
  {{"action": "tool_name", "args": {{"arg_name": "value"}}}}
- To give a final answer, output ONLY this JSON:
  {{"action": "final_answer", "answer": "your answer here"}}

RULES:
1. Output ONE JSON object per response — no prose, no markdown, no code fences.
2. Use a tool when you need to compute a value or look up information.
3. After receiving a tool result, decide: call another tool or give the final_answer.
4. If a tool returns an error, include it clearly in the final_answer.
5. Never invent a tool name — use only the tools listed below.

Available tools:{tools_section}"""


class ToolAgent(AgentBase):
    def __init__(
        self,
        llm_client: LlmClient,
        executor: ToolExecutor,
        config: ReActConfig = ReActConfig(),
    ) -> None:
        self._llm = llm_client
        self._executor = executor
        self._config = config

    def chat(self, user_input: str) -> str:
        self._executor.clear_traces()
        system = SystemMessage(content=_build_system_prompt(self._executor.tool_schemas()))
        messages = [system, HumanMessage(content=user_input)]

        for step in range(1, self._config.max_steps + 1):
            self._executor.log_trace(step, "PLAN", None, "LLM deciding next action...")
            raw = self._llm.invoke(messages)
            self._executor.log_trace(step, "PLAN", None, f"LLM output → {raw[:150]}")
            messages.append(AIMessage(content=raw))

            parsed = _parse_json(raw)
            if parsed is None:
                messages.append(HumanMessage(
                    content="Invalid format. Respond with ONLY a single JSON object — no prose, no markdown."
                ))
                repair = self._llm.invoke(messages)
                messages.append(AIMessage(content=repair))
                parsed = _parse_json(repair)
                if parsed is None:
                    return "I had trouble producing a valid response format. Please try rephrasing your question."

            action = parsed.get("action", "")
            if action == "final_answer":
                answer = str(parsed.get("answer", ""))
                if len(answer) > self._config.max_answer_length:
                    answer = answer[:self._config.max_answer_length] + " [truncated]"
                self._executor.log_trace(step, "OBSERVE", None, f"FINAL: {answer[:120]}")
                return answer

            result = self._executor.execute(step, action, parsed.get("args", {}))
            observation = (
                f"Tool '{action}' returned: {result.value}"
                if result.ok
                else f"Tool '{action}' failed: {result.error}. Report this error in your final_answer."
            )
            self._executor.log_trace(step, "OBSERVE", None, observation[:120])
            messages.append(HumanMessage(content=observation))

        return "Reached the maximum step limit without a final answer. Please try a simpler question."

    def reset(self) -> None:
        self._executor.clear_traces()
