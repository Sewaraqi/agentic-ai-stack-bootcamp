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


def load_long_term_memory(file_path: str) -> list[dict]:
    """Read the JSON memory store; return [] if missing or corrupt."""
    if not os.path.exists(file_path):
        print("> No memory store found — starting fresh.")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            sessions: list[dict] = json.load(f)
        print(f"> Loaded {len(sessions)} past session(s) from '{file_path}'")
        return sessions
    except json.JSONDecodeError:
        print(f"[WARN] Memory file '{file_path}' is corrupted — starting fresh.")
        return []


def save_long_term_memory(
    file_path: str,
    past_sessions: list[dict],
    session_id: str,
    history: list[BaseMessage],
) -> None:
    """Append the current session to the store and write to disk."""
    if not history:
        print("> Nothing to save (empty session).")
        return
    turns = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in history
    ]
    all_sessions = past_sessions + [{"session_id": session_id, "turns": turns}]
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, indent=2, ensure_ascii=False)
    print(f"> Session saved: {len(history) // 2} turn(s) → '{file_path}'")


def format_past_context(past_sessions: list[dict], max_sessions: int) -> str:
    """Return a readable block of the last N sessions to inject into the system prompt."""
    if not past_sessions:
        return ""
    recent = past_sessions[-max_sessions:]
    lines = [f"\n=== PAST SESSIONS (last {len(recent)} of {len(past_sessions)}) ==="]
    for session in recent:
        lines.append(f"\n[Session {session['session_id']}]")
        for turn in session.get("turns", []):
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"  {role}: {turn['content']}")
    lines.append("\n=== END OF PAST SESSIONS ===")
    return "\n".join(lines)


SESSION_ID = datetime.now().isoformat(timespec="seconds")
past_sessions = load_long_term_memory(MEMORY_FILE)
past_context = format_past_context(past_sessions, MAX_RECALLED_SESSIONS)

# Past sessions are injected once at startup into the system prompt.
# MessagesPlaceholder handles the current-session turns separately.
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

print(f"\n=== 03 - Long-Term Memory ===")
print(f"  Model         : {os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash')}")
print(f"  Memory file   : {MEMORY_FILE}")
print(f"  Past sessions : {len(past_sessions)}")
print(f"  Session ID    : {SESSION_ID}")
print("  Commands      : recall, history, save, exit\n")

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
        if user_input.lower() == "recall":
            print(past_context + "\n" if past_context else "  (no past sessions loaded)\n")
            continue
        if user_input.lower() == "save":
            save_long_term_memory(MEMORY_FILE, past_sessions, SESSION_ID, history)
            continue
        if user_input.lower() == "history":
            if not history:
                print("  (no history yet this session)\n")
            else:
                for i in range(0, len(history), 2):
                    u = history[i].content
                    a = history[i + 1].content if i + 1 < len(history) else ""
                    print(f"  [{i // 2 + 1}] You:   {u}")
                    print(f"       Agent: {a[:100]}{'...' if len(a) > 100 else ''}")
                print()
            continue

        response = chain.invoke({"history": history, "question": user_input})
        print(f"\nAgent: {response}\n")

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))

except KeyboardInterrupt:
    print("\nInterrupted.")

save_long_term_memory(MEMORY_FILE, past_sessions, SESSION_ID, history)
print("Goodbye!")
