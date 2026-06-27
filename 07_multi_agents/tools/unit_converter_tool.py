from base.tool_base import ToolBase, ToolResult, ToolSchema

# Each supported (from_unit, to_unit) pair maps to a function value -> result.
# Linear conversions use a factor; temperature needs an affine formula.
_CONVERSIONS = {
    ("km", "miles"): lambda v: v * 0.621371,
    ("miles", "km"): lambda v: v / 0.621371,
    ("kg", "lbs"): lambda v: v * 2.20462,
    ("lbs", "kg"): lambda v: v / 2.20462,
    ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
}


class UnitConverterTool(ToolBase):
    """
    Converts a numeric value from one unit to another.

    Supported pairs: km<->miles, kg<->lbs, celsius<->fahrenheit.
    Pure function — the same inputs always produce the same output, so
    is_idempotent=True and the call is safe to retry.
    """

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="unit_converter",
            description=(
                "Converts a numeric value from one unit to another. "
                "Supported conversions: km<->miles, kg<->lbs, Celsius<->Fahrenheit. "
                "Use this whenever the user asks to convert a distance, weight, or temperature."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The numeric value to convert.",
                    },
                    "from_unit": {
                        "type": "string",
                        "description": 'Source unit (e.g. "km", "Celsius").',
                    },
                    "to_unit": {
                        "type": "string",
                        "description": 'Target unit (e.g. "miles", "Fahrenheit").',
                    },
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        )

    def run(self, value: float, from_unit: str, to_unit: str) -> ToolResult:
        key = (from_unit.strip().lower(), to_unit.strip().lower())
        convert = _CONVERSIONS.get(key)
        if convert is None:
            supported = ", ".join(f"{f}->{t}" for f, t in _CONVERSIONS)
            return ToolResult(
                error=(
                    f"Unsupported conversion '{from_unit}' to '{to_unit}'. "
                    f"Supported pairs: {supported}."
                ),
                is_idempotent=True,
            )

        result = round(convert(value), 2)
        return ToolResult(value=f"{result} {key[1]}", is_idempotent=True)
