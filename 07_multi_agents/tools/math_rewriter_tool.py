import re

from langchain_core.messages import HumanMessage, SystemMessage

from base.tool_base import ToolBase, ToolResult, ToolSchema
from services.llm_client import LlmClient

# Sentinel the LLM returns when the input has no mathematical intent.
_NO_MATH = "NONE"

# A valid rewrite contains only digits, the allowed operators, parentheses,
# decimal points, and spaces. Anything else means the LLM did not produce a
# plain arithmetic string and we treat it as a failed rewrite.
_ARITHMETIC_RE = re.compile(r"^[\d+\-*/().\s]+$")


class MathRewriterTool(ToolBase):
    """
    Concept: Compound Tool — Knowledge Integration and Data Handling

    Translates a natural-language math expression ("a dozen plus a score")
    into a plain arithmetic string ("12 + 20") that CalculatorTool can evaluate.

    This is a compound tool — it calls the LLM internally (via LlmClient
    injection), exactly like QueryRewriterTool. The LLM is the translation
    engine; the tool is the contract wrapper that validates the output.

    Once registered in math_agent's executor BEFORE CalculatorTool, the
    existing ReAct loop calls math_rewriter first when the input contains
    words, then passes the arithmetic string to calculator — no change to
    the agent or the loop is needed.

    Idempotency: True — the same expression at low temperature always produces
    the same arithmetic form, so the call is safe to retry.
    """

    def __init__(self, llm_client: LlmClient) -> None:
        self._llm = llm_client

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="math_rewriter",
            description=(
                "Rewrites a natural-language math expression into a plain arithmetic "
                "string using only numbers and the operators + - * / ** and parentheses. "
                "Use this BEFORE calling the calculator whenever the math question "
                "contains words instead of digits, e.g. 'a dozen plus a score', "
                "'fifteen percent of two hundred', or 'half of sixty'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The natural-language math expression to rewrite.",
                    },
                },
                "required": ["expression"],
            },
        )

    def run(self, expression: str) -> ToolResult:
        if not expression.strip():
            return ToolResult(
                error="Cannot identify a mathematical expression in the input.",
                is_idempotent=True,
            )

        system = SystemMessage(content=(
            "You translate a natural-language math expression into a plain "
            "arithmetic string.\n"
            "Output ONLY the arithmetic expression — no explanation, no words, "
            "no surrounding text, no equals sign, no result.\n"
            "Use only digits, the operators + - * / **, parentheses, decimal "
            "points, and spaces.\n"
            "Convert number words to digits (a dozen -> 12, a score -> 20).\n"
            "Convert 'N percent of M' to 'M * 0.NN' (fifteen percent of two "
            "hundred -> 200 * 0.15).\n"
            "Convert 'a fraction of N' to division (half of sixty -> 60 / 2, "
            "a quarter of forty -> 40 / 4).\n"
            "Convert 'squared' to ** 2 and 'cubed' to ** 3.\n"
            f"If the input contains NO mathematical intent, output exactly: {_NO_MATH}"
        ))

        rewritten = self._llm.invoke(
            [system, HumanMessage(content=expression)]
        ).strip()

        # Strip code fences / backticks the LLM may add despite instructions.
        rewritten = rewritten.strip("`").strip()

        if not rewritten or rewritten.upper() == _NO_MATH or not _ARITHMETIC_RE.match(rewritten):
            return ToolResult(
                error="Cannot identify a mathematical expression in the input.",
                is_idempotent=True,
            )

        return ToolResult(value=rewritten, is_idempotent=True)
