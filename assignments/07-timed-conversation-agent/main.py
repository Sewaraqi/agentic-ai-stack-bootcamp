import os
import sys
from dotenv import load_dotenv
from agents.times_agent import TimedAgent
from services.llm_client import LlmConfig, LlmClient

load_dotenv()

config = LlmConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
)

print("=== Timed Conversation Agent ===")
print("Commands: stats, reset, exit\n")

with TimedAgent(LlmClient(config)) as agent:
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            sys.exit(0)
        if user_input.lower() == "stats":
            print(agent.stats())
            continue
        if user_input.lower() == "reset":
            agent.reset()
            print("Session cleared.")
            continue
        if user_input.lower() == "history":
            print(agent.history() + "\n")
            continue
        print(f"\nAgent: {agent.chat(user_input)}\n")
