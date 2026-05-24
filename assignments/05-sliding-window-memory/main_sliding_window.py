"""
Assignment 05 — Sliding Window Memory
Adds a configurable sliding window on top of the long-term memory pattern.
When history exceeds MAX_HISTORY_TURNS, the oldest turn is dropped,
keeping the context window from growing unboundedly.

Compare with main_bounded.py which uses a hard cap (fixed list size).
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "4"))


def trim_window(history: list[BaseMessage], max_turns: int) -> list[BaseMessage]:
    """Drop the oldest turn(s) until history holds at most max_turns turns.

    A turn is one HumanMessage + one AIMessage, so the message cap is
    max_turns * 2. Popping in pairs preserves user/assistant alternation.
    """
    while len(history) > max_turns * 2:
        history.pop(0)  # oldest HumanMessage
        history.pop(0)  # oldest AIMessage
    return history


system_message = "You are a helpful AI assistant specializing in agentic AI systems."

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
    api_key=api_key,
)
chain = prompt | llm | StrOutputParser()
history: list[BaseMessage] = []

print(f"\n=== 05 - Sliding Window Memory ===")
print(f"  Window size : {MAX_HISTORY_TURNS} turns (oldest drops when full)")
print("  Commands    : history, exit\n")

try:
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "history":
            current_turns = len(history) // 2
            print(f"  Window: {current_turns}/{MAX_HISTORY_TURNS} turns")
            for i in range(0, len(history), 2):
                print(f"  [{i // 2 + 1}] You: {history[i].content[:80]}")
            print()
            continue

        current_turns = len(history) // 2
        messages_sent = len(history) + 1
        print(f"  [sending {messages_sent} messages | window: {current_turns}/{MAX_HISTORY_TURNS}]")

        response = chain.invoke({"history": history, "question": user_input})
        print(f"\nAgent: {response}\n")

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))

        # Sliding window: drop the oldest turn when the window is full.
        trim_window(history, MAX_HISTORY_TURNS)
        assert len(history) <= MAX_HISTORY_TURNS * 2, \
            f"window overflow: {len(history)} messages > {MAX_HISTORY_TURNS * 2}"

except KeyboardInterrupt:
    print("\nInterrupted.")

print("Goodbye!")
