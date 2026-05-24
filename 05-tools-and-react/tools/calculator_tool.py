from base.tool_base import ToolBase, ToolResult, ToolSchema

_ALLOWED_OPS = {"add", "subtract", "multiply", "divide"}


class CalculatorTool(ToolBase):

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="calculator",
            description=(
                "Performs basic arithmetic on two numbers. "
                "Use this whenever the user asks to calculate, compute, multiply, add, subtract, or divide."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": sorted(_ALLOWED_OPS),
                        "description": "The arithmetic operation: add, subtract, multiply, divide.",
                    },
                    "a": {"type": "number", "description": "The first operand."},
                    "b": {"type": "number", "description": "The second operand."},
                },
                "required": ["operation", "a", "b"],
            },
        )

    def run(self, operation: str, a: float, b: float) -> ToolResult:
        if operation not in _ALLOWED_OPS:
            return ToolResult(error=f"Unknown operation '{operation}'. Allowed: {sorted(_ALLOWED_OPS)}")
        if operation == "divide" and b == 0:
            return ToolResult(error="Division by zero is not allowed.")
        result = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}[operation]
        return ToolResult(value=result)
