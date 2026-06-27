from langchain_core.messages import HumanMessage, SystemMessage

from base.tool_base import ToolBase, ToolResult, ToolSchema
from services.llm_client import LlmClient


class QueryRewriterTool(ToolBase):
    """
    Concept: Query Rewriting — Knowledge Integration and Data Handling

    Rewrites a vague or ambiguous user query into a precise, retrieval-optimized
    search query before passing it to a knowledge base or search tool.

    Why rewriting matters:
        A user asking "what about the weather?" after discussing London leaves the
        city implicit. A retrieval system needs the explicit form: "current weather
        in London". Without rewriting, the retrieval fails or returns irrelevant chunks.

    This is a compound tool — it calls the LLM internally (via LlmClient injection).
    That makes it different from CalculatorTool (pure function) and WeatherTool
    (mock lookup). The LLM is the rewriting engine; the tool is the contract wrapper.

    Idempotency: True — the same vague query with the same context always produces
    the same rewrite (given low temperature). Safe to retry if the LLM call fails.
    """

    def __init__(self, llm_client: LlmClient) -> None:
        self._llm = llm_client

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="query_rewriter",
            description=(
                "Rewrites a vague or ambiguous user query into a precise, "
                "retrieval-optimised search query. "
                "Use this BEFORE calling a search or lookup tool when the user's "
                "question is unclear, very short, or missing key details like a city name."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The original vague user query to rewrite.",
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional domain context to guide rewriting, "
                            "e.g. 'weather lookup' or 'city name required'."
                        ),
                    },
                },
                "required": ["query"],
            },
        )

    def run(self, query: str, context: str = "") -> ToolResult:
        if not query.strip():
            return ToolResult(error="Query cannot be empty.", is_idempotent=True)

        context_hint = f" Domain context: {context}." if context.strip() else ""
        system = SystemMessage(content=(
            f"You are a query rewriter for a retrieval system.{context_hint} "
            "Given a vague or short user query, rewrite it as a specific, "
            "self-contained search query that will retrieve the most relevant results. "
            "Output ONLY the rewritten query — no explanation, no punctuation changes."
        ))
        rewritten = self._llm.invoke([system, HumanMessage(content=query)])
        return ToolResult(value=rewritten.strip(), is_idempotent=True)
