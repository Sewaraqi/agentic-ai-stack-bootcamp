"""
Assignment 04 — Long-Term Memory Coach
Extends assignment 03 with cross-session persistence.
The user's answers and last draft are saved to data/introductions.json so the
model can pick up where it left off on the next run.
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

SAVE_FILE = "data/introductions.json"
SESSION_ID = datetime.now().isoformat(timespec="seconds")


def load_sessions() -> list[dict]:
    """Read saved sessions from disk; return [] if the file is missing or corrupt."""
    if not os.path.exists(SAVE_FILE):
        return []
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[WARN] '{SAVE_FILE}' is corrupted — starting fresh.")
        return []


def save_session(all_sessions: list[dict], record: dict) -> None:
    """Write the updated session list to disk, replacing any record with the same session_id."""
    updated = [s for s in all_sessions if s["session_id"] != record["session_id"]]
    updated.append(record)
    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    print(f"> Session saved → {SAVE_FILE}")


def collect_answers(full_name: str) -> dict:
    """Prompt the user for the 7 introduction questions and return them as a dict."""
    print()
    role        = input("Current role / degree: ").strip()
    experience  = input("Years of experience or seniority: ").strip()
    skills      = input("Top 3 skills (comma-separated): ").strip()
    achievement = input("One achievement you're proud of: ").strip()
    goal        = input("What you're looking for (goal): ").strip()
    fun_raw     = input("Fun fact (optional — press Enter to skip): ").strip()
    return {
        "name":        full_name,
        "role":        role,
        "experience":  experience,
        "skills":      skills,
        "achievement": achievement,
        "goal":        goal,
        "fun_fact":    fun_raw or "N/A",
    }


def make_tag(full_name: str) -> str:
    """Derive a school email tag from the user's full name."""
    parts = full_name.lower().split()
    return f"{parts[0]}.{parts[-1]}@school.edu" if len(parts) >= 2 else f"{parts[0]}@school.edu"


def history_to_turns(history: list[BaseMessage]) -> list[dict]:
    """Convert a LangChain message list to the plain-dict format used in JSON storage."""
    return [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in history
    ]


SESSION_ID = datetime.now().isoformat(timespec="seconds")
all_sessions = load_sessions()

print(f"\n=== 04 - Long-Term Memory Coach ===")
print(f"  Save file     : {SAVE_FILE}")
print(f"  Past sessions : {len(all_sessions)}  |  Session ID: {SESSION_ID}\n")

try:
    full_name = input("Full name: ").strip()
except KeyboardInterrupt:
    print("\nExiting.")
    sys.exit(0)

tag = make_tag(full_name)
user_sessions = [s for s in all_sessions if s.get("user", "").lower() == full_name.lower()]

answers = {}
history: list[BaseMessage] = []
latest_draft = ""

# Load the previous session or collect fresh answers
try:
    if user_sessions:
        last = user_sessions[-1]
        preview = last["last_draft"]
        print(f"\nFound your previous introduction (Session {last['session_id']}).")
        print(f"Last draft:\n  \"{preview[:120]}{'...' if len(preview) > 120 else ''}\"\n")
        choice = input("Continue refining? (yes / no): ").strip().lower()
        if choice == "yes":
            answers = last["answers"]
            for turn in last.get("turns", []):
                cls = HumanMessage if turn["role"] == "user" else AIMessage
                history.append(cls(content=turn["content"]))
            latest_draft = last["last_draft"]
        else:
            answers = collect_answers(full_name)
    else:
        answers = collect_answers(full_name)
except KeyboardInterrupt:
    print("\nExiting.")
    sys.exit(0)

# The last draft is injected into the system prompt so the model knows where to continue
past_draft_context = (
    f"\nThe user's most recent draft introduction is:\n\"{latest_draft}\"\n"
    "Continue refining from this draft unless the user asks to start over."
    if latest_draft else ""
)

system_message = (
    "You are a professional bio writer. Write a 120–150 word first-person introduction."
    + past_draft_context
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{current_input}"),
])

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
    api_key=api_key,
)
chain = prompt | llm | StrOutputParser()

# Generate the first draft only on a new session
if not history:
    fun = answers["fun_fact"] if answers["fun_fact"] != "N/A" else "N/A — skip this entirely"
    first_input = (
        f"Name: {answers['name']}\nRole: {answers['role']}\nExperience: {answers['experience']}\n"
        f"Skills: {answers['skills']}\nAchievement: {answers['achievement']}\n"
        f"Goal: {answers['goal']}\nFun fact: {fun}"
    )
    print(f"\n[Generating first draft — 1 message sent]")
    response = chain.invoke({"history": history, "current_input": first_input})
    print(f"\n--- First Draft ---\n\n{response.strip()}\n\nTAG: {tag}\n")
    history.append(HumanMessage(content=first_input))
    history.append(AIMessage(content=response))
    latest_draft = response.strip()

print("You can now refine the introduction. Type 'done' to finish.")
print("  Examples: 'make it less formal', 'add leadership experience'\n")


def build_record() -> dict:
    """Assemble the current session into the JSON schema expected by data/introductions.json."""
    return {
        "session_id": SESSION_ID,
        "user":       full_name,
        "answers":    answers,
        "turns":      history_to_turns(history),
        "last_draft": latest_draft,
    }


try:
    while True:
        try:
            user_input = input("Refinement (or 'done' to finish): ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() == "done":
            break

        response = chain.invoke({"history": history, "current_input": user_input})
        print(f"\n{response.strip()}\nTAG: {tag}\n")
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))
        latest_draft = response.strip()

except KeyboardInterrupt:
    print("\nInterrupted.")

save_session(all_sessions, build_record())

print("\n=== Final Introduction ===\n")
print(latest_draft)
print(f"\nSESSION: {SESSION_ID}")
print(f"TAG: {tag}")
