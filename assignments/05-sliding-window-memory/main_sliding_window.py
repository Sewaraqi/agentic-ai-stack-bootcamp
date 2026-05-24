"""
Assignment 05 — Sliding Window Memory
Adds a configurable sliding window on top of the long-term memory pattern.
When history exceeds MAX_HISTORY_TURNS, the oldest turn is dropped,
keeping the context window from growing unboundedly.

Compare with main_bounded.py which uses a hard cap (fixed list size).
"""
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")

MEMORY_FILE = os.getenv("MEMORY_FILE", "data/memory_store.json")
MAX_RECALLED_SESSIONS = int(os.getenv("MAX_RECALLED_SESSIONS", "3"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "4"))


def load_long_term_memory(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def save_long_term_memory(
    file_path: str, past_sessions: list[dict], session_id: str, history: list[BaseMessage]
) -> None:
    if not history:
        return
    turns = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in history
    ]
    all_sessions = past_sessions + [{"session_id": session_id, "turns": turns}]
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, indent=2, ensure_ascii=False)


def format_past_context(past_sessions: list[dict], max_sessions: int) -> str:
    if not past_sessions:
        return ""
    recent = past_sessions[-max_sessions:]
    lines = [f"\n=== PAST SESSIONS (last {len(recent)}) ==="]
    for session in recent:
        lines.append(f"\n[Session {session['session_id']}]")
        for turn in session.get("turns", []):
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"  {role}: {turn['content']}")
    lines.append("\n=== END ===")
    return "\n".join(lines)


SESSION_ID = datetime.now().isoformat(timespec="seconds")
past_sessions = load_long_term_memory(MEMORY_FILE)
past_context = format_past_context(past_sessions, MAX_RECALLED_SESSIONS)
system_message = "You are a helpful AI assistant specializing in agentic AI systems." + past_context

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
print(f"  Window size   : {MAX_HISTORY_TURNS} turns (oldest drops when full)")
print(f"  Past sessions : {len(past_sessions)}  |  Session ID: {SESSION_ID}")
print("  Commands      : history, exit\n")

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
                print(f"  [{i // 2 + 1}] You:   {history[i].content[:80]}")
            print()
            continue

        current_turns = len(history) // 2
        print(f"  [window: {current_turns}/{MAX_HISTORY_TURNS}]")

        response = chain.invoke({"history": history, "question": user_input})
        print(f"\nAgent: {response}\n")

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))

        # Drop the oldest turn when the window is full
        if len(history) > MAX_HISTORY_TURNS * 2:
            history.pop(0)  # oldest HumanMessage
            history.pop(0)  # oldest AIMessage

except KeyboardInterrupt:
    print("\nInterrupted.")

save_long_term_memory(MEMORY_FILE, past_sessions, SESSION_ID, history)
print("Goodbye!")
