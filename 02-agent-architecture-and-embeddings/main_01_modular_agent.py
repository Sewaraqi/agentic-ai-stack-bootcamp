import os
import sys
from dotenv import load_dotenv
from agents.conversation_agent import ConversationAgent
from services.llm_client import LlmConfig, LlmClient

load_dotenv()

config = LlmConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
)

print("=== 01 - Modular Agent Architecture ===")
print("Commands: reset, history, exit\n")

with ConversationAgent(LlmClient(config)) as agent:
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
        if user_input.lower() == "reset":
            agent.reset()
            print("[History cleared]\n")
            continue
        if user_input.lower() == "history":
            print(agent.history_text() + "\n")
            continue
        print(f"\nAgent: {agent.chat(user_input)}\n")
