import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from agents.tool_agent import ToolAgent, ReActConfig
from services.llm_client import LlmClient, LlmConfig
from services.tool_executor import ToolExecutor
from tools.calculator_tool import CalculatorTool
from tools.weather_tool import WeatherTool

load_dotenv(override=True)

llm = LlmClient(LlmConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.1")),
))

executor = ToolExecutor(max_retries=2, base_delay=0.5)
executor.register(CalculatorTool())
executor.register(WeatherTool())

agent = ToolAgent(llm, executor, ReActConfig(max_steps=6))

print("=== 01 - Tool Agent (ReAct Loop) ===\n")
print("Registered tools:")
for schema in executor.tool_schemas():
    print(f"  {schema['name']:<16}  {schema['description'][:65]}")
print()
print("Try these queries:")
print("  'What is 42 multiplied by 13?'")
print("  'What is the weather in Tel Aviv?'")
print("  'What is (15 + 7) multiplied by 2?'     ← two tool calls")
print("  'What is the weather on the moon?'       ← error: unknown city")
print("  'What is 10 divided by 0?'               ← error: business rule")
print()
print("Commands: trace (show last query's Plan/Act/Observe log) | exit\n")

with agent:
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            sys.exit(0)
        if user_input.lower() == "trace":
            traces = executor.get_traces()
            if not traces:
                print("No traces yet.\n")
                continue
            print("\n[Plan/Act/Observe trace]")
            for t in traces:
                tool_col = f"[{t.tool_name}]" if t.tool_name else "           "
                ms_str = f"  ({t.duration:.0f} ms)" if t.duration > 0 else ""
                print(f"   step {t.step}  {t.phase:<8}  {tool_col:<16}  {t.details[:90]}{ms_str}")
            print()
            continue
        answer = agent.chat(user_input)
        print(f"\nAgent: {answer}\n")
