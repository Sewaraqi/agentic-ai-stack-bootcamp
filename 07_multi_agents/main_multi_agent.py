"""
╔══════════════════════════════════════════════════════════════════╗
║  tutorial_07 — main_multi_agent.py                           ║
║  NCP-AAI Blueprint: Agent Architecture and Design (15%)         ║
║                     Knowledge Integration and Data Handling(10%)║
╠══════════════════════════════════════════════════════════════════╣
║  CONCEPT        Router pattern  → one specialist per intent     ║
║                 Orchestrator    → decompose → fan-out → synth   ║
║  PROBLEM SOLVED Tasks that span multiple specialist domains     ║
╚══════════════════════════════════════════════════════════════════╝
"""
import os
import sys

from dotenv import load_dotenv

from agents.dispatcher_agent import DispatcherAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.router_agent import RouterAgent
from agents.specialist_agent import SpecialistAgent, SpecialistConfig
from services.agent_registry import AgentRegistry
from services.llm_client import LlmClient, LlmConfig
from services.tool_executor import ToolExecutor
from tools.calculator_tool import CalculatorTool
from tools.query_rewriter_tool import QueryRewriterTool
from tools.weather_tool import WeatherTool

load_dotenv(override=True)

# Shared LLM client
llm = LlmClient(LlmConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_MODEL_NAME"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.1")),
))

# Build specialists

# math_agent: only arithmetic — no weather tools in scope
math_executor = ToolExecutor(max_retries=2, base_delay=0.5)
math_executor.register(CalculatorTool())

math_agent = SpecialistAgent(
    llm_client=llm,
    executor=math_executor,
    config=SpecialistConfig(
        role="math_agent",
        description="Handles arithmetic calculations, number computations, and math questions.",
        max_steps=4,
    ),
)

# weather_agent: weather lookup + query rewriter for vague city queries
weather_executor = ToolExecutor(max_retries=1, base_delay=0.5)
weather_executor.register(QueryRewriterTool(llm))
weather_executor.register(WeatherTool())

weather_agent = SpecialistAgent(
    llm_client=llm,
    executor=weather_executor,
    config=SpecialistConfig(
        role="weather_agent",
        description=(
            "Handles weather queries, temperature lookups, and climate questions "
            "for specific cities. Can rewrite vague queries to extract the city name."
        ),
        max_steps=5,
        system_hint=(
            "When the current user message does not contain an explicit city name, "
            "you MUST call query_rewriter before calling weather. "
            "Pass the full user message as 'query' and 'weather lookup' as 'context'."
        ),
    ),
)

# general_agent: no tools — handles conversational questions with LLM knowledge
general_executor = ToolExecutor()

general_agent = SpecialistAgent(
    llm_client=llm,
    executor=general_executor,
    config=SpecialistConfig(
        role="general_agent",
        description=(
            "Handles general knowledge questions, explanations, definitions, "
            "and conversational questions that do not require computation or weather data."
        ),
        max_steps=3,
    ),
)

# Build registry
registry = AgentRegistry()
registry.register(math_agent)
registry.register(weather_agent)
registry.register(general_agent)

# Build router and orchestrator
router = RouterAgent(llm_client=llm, registry=registry, max_hops=3)
orchestrator = OrchestratorAgent(llm_client=llm, registry=registry, max_subtasks=5)
# Dispatcher: classifies every query and delegates to router or orchestrator
dispatcher = DispatcherAgent(
    llm_client=llm,
    registry=registry,
    router=router,
    orchestrator=orchestrator,
)

# Conversation history — passed to every dispatcher.chat() call
history: list[dict] = []
# Conversation history buffer
# Each entry: {"role": "user"|"assistant", "content": str}
# Passed to every agent call so vague follow-ups resolve from prior context.
history: list[dict] = []

# Startup banner
print("=== tutorial_07 — Multi-Agent Orchestration ===\n")
print("Registered specialists:")
for role, desc in registry.descriptions().items():
    print(f"  {role:<20}  {desc[:65]}")
print()
print("Commands:")
print("  subtasks                         — results from last multi-domain query")
print("  trace <math|weather|general>   — step trace for a specialist")
print("  history                          — show conversation history")
print("  clear                            — clear conversation history")
print("  exit                             — quit")
print()
print("Just type any question — routing is automatic.")
print()
print("Example queries (try in order to see history in action):")
print("  What is the weather in London?")
print("  What about there?")
print("  What is 42 multiplied by 13?")
print("  What is the capital of France?")
print("  Compare the weather in London and Tel Aviv and calculate the temperature difference.")
print()

_specialist_map = {
    "math": (math_agent, math_executor),
    "weather": (weather_agent, weather_executor),
    "general": (general_agent, general_executor),
}

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        sys.exit(0)

    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        sys.exit(0)

    # history command
    if user_input.lower() == "history":
        if not history:
            print("[History is empty]\n")
        else:
            print(f"\n[Conversation history — {len(history)} turns]")
            for i, turn in enumerate(history, 1):
                prefix = "You:   " if turn["role"] == "user" else "Agent: "
                print(f"  {i:2}. {prefix}{turn['content'][:100]}")
            print()
        continue

    # clear command
    if user_input.lower() == "clear":
        history.clear()
        print("[History cleared]\n")
        continue

    # subtasks command
    if user_input.lower() == "subtasks":
        results = orchestrator.get_subtask_results()
        if not results:
            print("[No multi-domain query yet]\n")
            continue
        print("\n[Subtask results from last multi-domain query]")
        for i, r in enumerate(results, 1):
            print(f"  Subtask {i}: [{r['role']}] {r['task'][:60]}")
            print(f"    → {r['result'][:100]}")
        print()
        continue

    # trace command
    if user_input.lower().startswith("trace"):
        parts = user_input.split()
        key = parts[1].lower() if len(parts) > 1 else ""
        if key not in _specialist_map:
            print(f"Usage: trace <{'|'.join(_specialist_map.keys())}>\n")
            continue
        _, executor = _specialist_map[key]
        traces = executor.get_traces()
        if not traces:
            print(f"[No trace yet for {key}]\n")
            continue
        print(f"\n[Plan / Act / Observe trace — {key}]")
        for t in traces:
            tool_col = f"[{t.tool_name}]" if t.tool_name else "         "
            ms_str = f"  ({t.duration:.0f} ms)" if t.duration > 0 else ""
            print(
                f"  step {t.step}  {t.phase:<8}  "
                f"{tool_col:<20}  {t.details[:80]}{ms_str}"
            )
        print()
        continue
    # dispatcher: handles all free-form queries
    print(f"\n[Classifying...]\n")
    answer = dispatcher.chat(user_input, history)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": answer})
    print(f"  Mode: {dispatcher.last_mode()}")
    print(f"\nAgent: {answer}\n")

    # # route command
    # if user_input.lower().startswith("route "):
    #     question = user_input[6:].strip()
    #     if not question:
    #         print("Usage: route <question>\n")
    #         continue
    #     print(f"\n[Routing...]\n")
    #     answer = router.chat(question, history)
    #     history.append({"role": "user", "content": question})
    #     history.append({"role": "assistant", "content": answer})
    #     print(f"  Routed to: {router.last_route()}")
    #     print(f"\nAgent: {answer}\n")
    #     continue
    #
    # # orchestrate command
    # if user_input.lower().startswith("orchestrate "):
    #     task = user_input[12:].strip()
    #     if not task:
    #         print("Usage: orchestrate <task>\n")
    #         continue
    #     print(f"\n[Decomposing and executing subtasks...]\n")
    #     answer = orchestrator.chat(task, history)
    #     history.append({"role": "user", "content": task})
    #     history.append({"role": "assistant", "content": answer})
    #     print(f"\nAgent: {answer}\n")
    #     print("  (type 'subtasks' to see per-specialist results)\n")
    #     continue
    # print("Unknown command. Use 'route', 'orchestrate', 'subtasks', 'trace', 'history', 'clear', or 'exit'.\n")
