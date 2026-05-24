"""
Assignment 4 — Ask-then-Introduce (Long-Term Memory)
Persists conversation history to data/introductions.json so sessions survive
process restarts. On first run: collects answers and generates a draft.
On subsequent runs: loads the last draft and continues refinement.
"""

import os
import sys
import json
import signal
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME  = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
SAVE_PATH          = "data/introductions.json"

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and set your key.")

# ─────────────────────────────────────────────
# 1. Load past sessions from JSON
#    Returns [] if the file doesn't exist yet.
# ─────────────────────────────────────────────
def load_sessions() -> list[dict]:
    if not os.path.exists(SAVE_PATH):
        return []
    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# 5. Save session to JSON on exit
#    Appends the new record then writes the
#    full list back to disk atomically.
# ─────────────────────────────────────────────
def save_session(sessions: list[dict], record: dict) -> None:
    os.makedirs("data", exist_ok=True)
    # Replace existing record for same session_id, or append new one
    updated = [s for s in sessions if s["session_id"] != record["session_id"]]
    updated.append(record)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    print(f"\n> Session saved → {SAVE_PATH}")


# ─────────────────────────────────────────────
# Helper — derive TAG from full name
# ─────────────────────────────────────────────
def make_tag(full_name: str) -> str:
    parts = full_name.lower().split()
    return f"{parts[0]}.{parts[-1]}@school.edu"


# ─────────────────────────────────────────────
# Helper — rebuild LangChain message history
#    from the saved turns list so the model
#    gets full context on subsequent sessions.
# ─────────────────────────────────────────────
def turns_to_messages(turns: list[dict]) -> list[BaseMessage]:
    messages = []
    for turn in turns:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages


# ─────────────────────────────────────────────
# Build LLM + chain (reused across all turns)
# ─────────────────────────────────────────────
def build_chain(system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{current_input}"),
    ])
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        temperature=GEMINI_TEMPERATURE,
        api_key=GEMINI_API_KEY,
    )
    return prompt | llm | StrOutputParser()


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

# ─────────────────────────────────────────────
# 1. Load all past sessions
# ─────────────────────────────────────────────
all_sessions = load_sessions()

# ─────────────────────────────────────────────
# 2. Ask for the user's name first so we can
#    look up their past sessions
# ─────────────────────────────────────────────
print()
full_name = input("Full name: ").strip()
tag       = make_tag(full_name)

# Find the most recent past session for this user (case-insensitive)
past_sessions = [
    s for s in all_sessions
    if s["user"].lower() == full_name.lower()
]
past_session = past_sessions[-1] if past_sessions else None

# Generate a new session ID for this run
session_id = datetime.now().isoformat(timespec="seconds")

# State we'll build up during this run
answers:     dict           = {}
history:     list[BaseMessage] = []
turns:       list[dict]     = []
latest_draft: str           = ""

# ─────────────────────────────────────────────
# 3a. Past session found — show last draft,
#     ask the user whether to continue or
#     start fresh.
# ─────────────────────────────────────────────
if past_session:
    print(f"\n> Loaded past session for {full_name} ({past_session['session_id']})")
    print(f"\nLast draft:\n  {past_session['last_draft']}\n")

    choice = input("Continue refining? (yes / no): ").strip().lower()

    if choice == "yes":
        # Restore answers and history from the saved session
        answers      = past_session["answers"]
        history      = turns_to_messages(past_session["turns"])
        latest_draft = past_session["last_draft"]
        turns        = list(past_session["turns"])  # copy so we can append

        # System prompt reminds the model of the previous draft context
        system_message = (
            "You are a professional bio writer and helpful assistant. "
            "The user has an existing introduction draft shown in the conversation history. "
            "When asked to refine, update the most recent draft according to the instruction "
            "while keeping it 120–150 words unless the user explicitly asks for a different length. "
            "Output ONLY the paragraph — no headings, no preamble."
        )

    else:
        # Start fresh — fall through to the new-user flow below
        past_session = None

# ─────────────────────────────────────────────
# 3b. No past session (or user chose fresh):
#     collect all 7 answers and generate the
#     first draft.
# ─────────────────────────────────────────────
if not past_session:
    print("\nPlease answer the following questions.\n")
    role         = input("2. Current role / degree: ").strip()
    experience   = input("3. Years of experience or seniority: ").strip()
    skills       = input("4. Top 3 skills (comma-separated): ").strip()
    achievement  = input("5. One achievement you're proud of: ").strip()
    goal         = input("6. What you're looking for (goal): ").strip()
    fun_fact_raw = input("7. Fun fact (optional — press Enter to skip): ").strip()
    fun_fact     = fun_fact_raw if fun_fact_raw else "N/A"

    answers = {
        "name":        full_name,
        "role":        role,
        "experience":  experience,
        "skills":      skills,
        "achievement": achievement,
        "goal":        goal,
        "fun_fact":    fun_fact,
    }

    system_message = (
        "You are a professional bio writer and helpful assistant. "
        "When given a person's details, write a first-person introduction "
        "paragraph that is STRICTLY 120–150 words. "
        "When asked to refine, update the most recent draft according to the "
        "instruction while keeping it 120–150 words unless told otherwise. "
        "Output ONLY the paragraph — no headings, no preamble."
    )

    first_human_message = (
        f"Name: {full_name}\n"
        f"Role / Degree: {answers['role']}\n"
        f"Experience / Seniority: {answers['experience']}\n"
        f"Top 3 Skills: {answers['skills']}\n"
        f"Achievement: {answers['achievement']}\n"
        f"Goal: {answers['goal']}\n"
        f"Fun fact: {answers['fun_fact']}"
    )

    chain = build_chain(system_message)

    print(f"\n[sending {len(history) + 1} messages]")
    latest_draft = chain.invoke({
        "history":       history,
        "current_input": first_human_message,
    })

    print("\n--- First Draft ---\n")
    print(latest_draft.strip())
    print(f"\nTAG: {tag}")

    # Save turn 1 into both history (LangChain objects) and turns (JSON-serialisable)
    history.append(HumanMessage(content=first_human_message))
    history.append(AIMessage(content=latest_draft))
    turns.append({"role": "user",      "content": first_human_message})
    turns.append({"role": "assistant", "content": latest_draft})

# Build the chain (for continuing sessions the chain is built here)
chain = build_chain(system_message)

# ─────────────────────────────────────────────
# Shared save helper used by both clean exit
# and Ctrl-C so we never lose a session.
# ─────────────────────────────────────────────
def do_save():
    record = {
        "session_id": session_id,
        "user":       full_name,
        "answers":    answers,
        "turns":      turns,
        "last_draft": latest_draft.strip(),
    }
    save_session(all_sessions, record)

# Catch Ctrl-C at the signal level so save always runs
def handle_sigint(sig, frame):
    print("\n\nInterrupted — saving session before exit.")
    do_save()
    print_final()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

def print_final():
    print("\n=== Final Introduction ===\n")
    print(latest_draft.strip())
    print(f"\nSESSION: {session_id}")
    print(f"TAG: {tag}")

# ─────────────────────────────────────────────
# 4. Refinement loop — identical logic to
#    Assignment 3 but turns are also written
#    to the JSON-serialisable turns list.
# ─────────────────────────────────────────────
while True:
    user_input = input("\nRefinement (or 'done' to finish): ").strip()

    if not user_input:
        continue

    if user_input.lower() == "done":
        break

    total_messages = 1 + len(history) + 1
    print(f"[sending {total_messages} messages]")

    refined = chain.invoke({
        "history":       history,
        "current_input": user_input,
    })

    print(f"\n{refined.strip()}")
    print(f"\nTAG: {tag}")

    # Append to both in-memory history and the serialisable turns list
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=refined))
    turns.append({"role": "user",      "content": user_input})
    turns.append({"role": "assistant", "content": refined})

    latest_draft = refined

# ─────────────────────────────────────────────
# 5 & 6. Clean exit: save then print final
# ─────────────────────────────────────────────
do_save()
print_final()