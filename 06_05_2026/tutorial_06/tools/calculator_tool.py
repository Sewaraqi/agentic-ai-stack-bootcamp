from base.tool_base import ToolBase, ToolResult, ToolSchema

_ALLOWED_OPS = {"add", "subtract", "multiply", "divide"}


class CalculatorTool(ToolBase):

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="Calculator",
            description=(
                "Performs basic arithmetic on two numbers."
                "Use this whenever the user asks you to calculate, compute,"
                "multiply, add, subtract, or divide numbers."),
            parameters={
                "type": "object",
                "properties":
                    {
                        "operation":
                            {
                                "type": "string",
                                "enum": sorted(_ALLOWED_OPS),
                                "description": "The operation: add, subtract, multiply, divide",
                            },
                        "a": {"type": "number", "description": "The first number.", },
                        "b": {"type": "number", "description": "The second number.", }, },
                "required": ["operation", "a", "b"], }, )

    def run(self, operation: str, a: float, b: float) -> ToolResult:
        if operation not in _ALLOWED_OPS:
            return ToolResult(
                error=f"unknown operation '{operation}'. Allowed {sorted(_ALLOWED_OPS)}",
                is_idempotent=True)
        if operation == "divide" and b == 0:
            return ToolResult(
                is_idempotent=True,
                error="Division by zero is not allowed.",
            )
        result = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b,
        }[operation]
        return ToolResult(value=result, is_idempotent=True)
