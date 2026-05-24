import re
from base.tool_base import ToolBase, ToolResult, ToolSchema

_MOCK_DATA: dict[str, str] = {
    "tel aviv": "Sunny, 28°C",
    "jerusalem": "Partly cloudy, 22°C",
    "haifa": "Windy, 24°C",
    "london": "Rainy, 12°C",
    "paris": "Overcast, 15°C",
    "new york": "Clear, 18°C",
    "tokyo": "Humid, 26°C",
}

_MAX_CITY_LEN = 50


class WeatherTool(ToolBase):

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="weather",
            description=(
                "Returns the current weather for a named city. "
                "Use this when the user asks about weather, temperature, or conditions in a specific city."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name in English, e.g. 'Tel Aviv' or 'London'.",
                        "maxLength": _MAX_CITY_LEN,
                    },
                },
                "required": ["city"],
            },
        )

    def run(self, city: str) -> ToolResult:
        if len(city) > _MAX_CITY_LEN:
            return ToolResult(error=f"City name too long (max {_MAX_CITY_LEN} characters).", is_idempotent=False)
        if not re.match(r"^[a-zA-Z\s\-]+$", city.strip()):
            return ToolResult(error="City name must contain only letters, spaces, or hyphens.", is_idempotent=False)
        weather = _MOCK_DATA.get(city.lower().strip())
        if weather is None:
            known = ", ".join(_MOCK_DATA.keys())
            return ToolResult(error=f"No weather data for '{city}'. Known cities: {known}.", is_idempotent=False)
        return ToolResult(value=weather, is_idempotent=False)
