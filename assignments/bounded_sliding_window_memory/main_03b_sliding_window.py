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

GEMINI_API_KEY: str = 'gemini_api_key'
GEMINI_MODEL_NAME: str = 'gemini_model_name'
GEMINI_TEMPERATURE: str = 'gemini_temperature'

# added for long-term memory
MEMORY_FILE: str = 'memory_file'
MAX_RECALLED_SESSIONS: str = 'max_recalled_sessions'

# added for sliding window
MAX_HISTORY_TURNS: str = 'max_history_turns'

ENV_GEMINI_API_KEY: str = 'GEMINI_API_KEY'
ENV_GEMINI_MODEL_NAME: str = 'GEMINI_MODEL_NAME'
ENV_GEMINI_TEMPERATURE: str = 'GEMINI_TEMPERATURE'

# added for long-term memory
ENV_MEMORY_FILE: str = 'MEMORY_FILE'
ENV_MAX_RECALLED_SESSIONS: str = 'MAX_RECALLED_SESSIONS'

# added for sliding window
ENV_MAX_HISTORY_TURNS: str = 'MAX_HISTORY_TURNS'

config: dict[str, object] = {
    GEMINI_API_KEY: os.getenv(ENV_GEMINI_API_KEY),
    GEMINI_MODEL_NAME: os.getenv(ENV_GEMINI_MODEL_NAME, 'gemini-1.5-flash'),
    GEMINI_TEMPERATURE: float(os.getenv(ENV_GEMINI_TEMPERATURE, '0.0')),
    # added for long-term memory
    MEMORY_FILE: os.getenv(ENV_MEMORY_FILE, 'data/memory_store.json'),
    MAX_RECALLED_SESSIONS: int(os.getenv(ENV_MAX_RECALLED_SESSIONS, '3')),
    # added for sliding window
    MAX_HISTORY_TURNS: int(os.getenv(ENV_MAX_HISTORY_TURNS, '4')),
}

if not config[GEMINI_API_KEY]:
    raise RuntimeError(f"{ENV_GEMINI_API_KEY} is not set, Copy .env.example to .env and set your key.")


# ---------------------------------------------------------------------------
# added for long-term memory: load past sessions from JSON file
# ---------------------------------------------------------------------------
def load_long_term_memory(file_path: str) -> list[dict]:
    """Read the JSON memory store; return [] if missing or corrupt."""
    if not os.path.exists(file_path):
        print("> No memory store found — starting fresh.")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sessions: list[dict] = json.load(f)
        print(f"> Long-term memory loaded: {len(sessions)} past session(s) from '{file_path}'")
        return sessions
    except json.JSONDecodeError:
        print(f"[WARN] Memory file '{file_path}' is corrupted — starting fresh.")
        return []


# ---------------------------------------------------------------------------
# added for long-term memory: save current session to JSON file
# ---------------------------------------------------------------------------
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
    turns: list[dict] = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant",
         "content": m.content}
        for m in history
    ]
    current_session: dict = {"session_id": session_id, "turns": turns}
    all_sessions: list[dict] = past_sessions + [current_session]
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_sessions, f, indent=2, ensure_ascii=False)
    print(f"> Session saved: {len(history) // 2} turn(s) → '{file_path}'")


# ---------------------------------------------------------------------------
# added for long-term memory: format past sessions as a text block
# ---------------------------------------------------------------------------
def format_past_context(past_sessions: list[dict], max_sessions: int) -> str:
    """Return a readable string of the last N sessions for system prompt injection."""
    if not past_sessions:
        return ""
    recent: list[dict] = past_sessions[-max_sessions:]
    lines: list[str] = [f"\n=== PAST SESSIONS (last {len(recent)} of {len(past_sessions)}) ==="]
    for session in recent:
        lines.append(f"\n[Session {session['session_id']}]")
        for turn in session.get("turns", []):
            role: str = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"  {role}: {turn['content']}")
    lines.append("\n=== END OF PAST SESSIONS ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# added for long-term memory: load past sessions and build context string
# ---------------------------------------------------------------------------
SESSION_ID: str = datetime.now().isoformat(timespec='seconds')
past_sessions: list[dict] = load_long_term_memory(config[MEMORY_FILE])
past_context_str: str = format_past_context(past_sessions, config[MAX_RECALLED_SESSIONS])

# ---------------------------------------------------------------------------
# added for long-term memory: system message includes past sessions text block.
# MessagesPlaceholder still handles the current-session short-term history.
# ---------------------------------------------------------------------------
system_message: str = (
        "You are a helpful AI assistant specialising in agentic AI systems."
        + past_context_str  # ← past sessions injected here at startup
)

prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ('system', system_message),
    MessagesPlaceholder(variable_name='history'),  # ← current session turns
    ('human', '{question}'),
])

chain = prompt_template | ChatGoogleGenerativeAI(
    model=config[GEMINI_MODEL_NAME],
    temperature=config[GEMINI_TEMPERATURE],
    api_key=config[GEMINI_API_KEY],
) | StrOutputParser()

# ---------------------------------------------------------------------------
# Short-term history for the current session (bounded by sliding window)
# ---------------------------------------------------------------------------
history: list[BaseMessage] = []

max_turns: int = config[MAX_HISTORY_TURNS]

print(f"\n=== Loop + Short-Term + Long-Term Memory + Sliding Window ===")
print(f"  Model          : {config[GEMINI_MODEL_NAME]}")
print(f"  Memory file    : {config[MEMORY_FILE]}")
print(f"  Past sessions  : {len(past_sessions)}")
print(f"  Session ID     : {SESSION_ID}")
print(f"  Window size    : {max_turns} turns")
print("  Commands       : 'recall', 'history', 'save', 'exit' / 'quit'\n")

try:
    while True:
        try:
            user_input: str = input("You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            break

        # added for long-term memory: show past context loaded from disk
        if user_input.lower() == "recall":
            print(past_context_str + "\n" if past_context_str else "  (no past sessions loaded)\n")
            continue

        # added for long-term memory: force save mid-session
        if user_input.lower() == "save":
            save_long_term_memory(config[MEMORY_FILE], past_sessions, SESSION_ID, history)
            continue

        if user_input.lower() == "history":
            if not history:
                print("  (no history yet this session)\n")
            else:
                print(f"\n  {len(history) // 2} turn(s) this session:")
                for i in range(0, len(history), 2):
                    u = history[i].content
                    a = history[i + 1].content if i + 1 < len(history) else ""
                    print(f"  [{i // 2 + 1}] You:   {u}")
                    print(f"       Agent: {a[:100]}{'...' if len(a) > 100 else ''}")
                print()
            continue

        current_turns: int = len(history) // 2
        print(f"  [sending {len(history) + 2} messages | window: {current_turns}/{max_turns}]")

        ai_response: str = chain.invoke({"history": history, "question": user_input})
        print(f"\nAgent: {ai_response}\n")

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=ai_response))

        # added for sliding window: drop oldest turn when window is full
        if len(history) > max_turns * 2:
            history.pop(0)  # drop oldest HumanMessage
            history.pop(0)  # drop oldest AIMessage

except KeyboardInterrupt:
    print("\nInterrupted.")

# added for long-term memory: always save before exiting
save_long_term_memory(config[MEMORY_FILE], past_sessions, SESSION_ID, history)
print("Goodbye!")
