"""
Assignment 06 — Token-Aware Chatbot

Tracks real token usage per turn via usage_metadata from the LLM response.
- Displays a token status line after every reply.
- Summarises the conversation automatically when remaining tokens fall
  below SUMMARY_THRESHOLD, replacing the history with a compact summary pair.
- Closes the session gracefully (no LLM call) when remaining tokens fall
  below MIN_RESPONSE_TOKENS.
- Short-term memory: history list + MessagesPlaceholder.
- Long-term memory: data/user_profile.json injected into the system message,
  so personal info survives history compression.
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --------------------------------------------------------------------------- #
# 0. LLM setup
# --------------------------------------------------------------------------- #
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
    api_key=api_key,
)

# --------------------------------------------------------------------------- #
# 1. Config — token budget, read from .env with defaults
# --------------------------------------------------------------------------- #
config = {
    "TOKEN_LIMIT": int(os.getenv("TOKEN_LIMIT", 4000)),
    "SUMMARY_THRESHOLD": int(os.getenv("SUMMARY_THRESHOLD", 800)),
    "MIN_RESPONSE_TOKENS": int(os.getenv("MIN_RESPONSE_TOKENS", 200)),
}

PROFILE_PATH = Path("data/user_profile.json")

# --------------------------------------------------------------------------- #
# 2. Long-term memory: load the user profile (or {} if none)
# --------------------------------------------------------------------------- #
def load_profile(file_path: Path) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_profile(profile: dict, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def profile_block(profile: dict) -> str:
    """Render the profile as a system-message block. Empty if no profile."""
    if not profile:
        return ""
    name = profile.get("name", "Unknown")
    role = profile.get("role", "Unknown")
    prefs = ", ".join(profile.get("preferences", [])) or "none"
    notes = ", ".join(profile.get("notes", [])) or "none"
    return (
        "\n=== USER PROFILE ===\n"
        f"Name: {name} | Role: {role} | "
        f"Preferences: {prefs} | Notes: {notes}\n"
        "=== END PROFILE ==="
    )


profile = load_profile(PROFILE_PATH)
if not profile:
    print("\n> No user profile found — starting fresh.")

# --------------------------------------------------------------------------- #
# 3. Prompt template with MessagesPlaceholder.
#    The system message = base role + injected profile, so it survives
#    summarisation (history is compressed, the system message is not).
# --------------------------------------------------------------------------- #
def build_prompt(profile: dict) -> ChatPromptTemplate:
    system_text = "You are a helpful assistant." + profile_block(profile)
    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])


prompt = build_prompt(profile)

# --------------------------------------------------------------------------- #
# 4. Token tracking state + short-term memory
# --------------------------------------------------------------------------- #
history: list[BaseMessage] = []
tokens_used_total: int = 0
tokens_remaining: int = config["TOKEN_LIMIT"]
last_in_tok: int = 0
last_out_tok: int = 0


# --------------------------------------------------------------------------- #
# 5. Helper: summarise_history -> (summary_text, summary_input_tokens, summary_output_tokens)
# --------------------------------------------------------------------------- #
def summarise_history(hist: list[BaseMessage]) -> tuple[str, int, int]:
    turns_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in hist
    )
    summary_messages = [
        SystemMessage(content="You are a memory compressor. Be concise and factual."),
        HumanMessage(content=f"Summarise this conversation in under 100 words:\n{turns_text}"),
    ]
    ai_msg: AIMessage = llm.invoke(summary_messages)
    meta = ai_msg.usage_metadata or {}
    # input_tokens ≈ the size of the history we're compressing away;
    # output_tokens ≈ the size of the compact summary that replaces it.
    return ai_msg.content, meta.get("input_tokens", 0), meta.get("output_tokens", 0)


# --------------------------------------------------------------------------- #
# 6. Helper: the "session closed" box + graceful exit
# --------------------------------------------------------------------------- #
def close_chat() -> None:
    save_profile(profile, PROFILE_PATH)
    print()
    print("╔══════════════════════════════════════════╗")
    print("║  Chat session closed.                    ║")
    print("║  Token limit reached for today.          ║")
    print("║  Your profile has been saved.            ║")
    print("║  Come back in a few hours!               ║")
    print("╚══════════════════════════════════════════╝")
    sys.exit(0)


def do_save_profile() -> None:
    """Prompt for name / role / one preference and persist to JSON."""
    name = input("  Name       : ").strip()
    role = input("  Role       : ").strip()
    pref = input("  Preference : ").strip()
    profile["name"] = name
    profile["role"] = role
    profile["preferences"] = [pref] if pref else []
    profile.setdefault("notes", [])
    save_profile(profile, PROFILE_PATH)
    # Rebuild the prompt so the freshly saved profile is injected immediately.
    global prompt
    prompt = build_prompt(profile)
    print(f"> Profile saved → {PROFILE_PATH}")


# --------------------------------------------------------------------------- #
# Startup banner
# --------------------------------------------------------------------------- #
print("\n=== Token-Aware Chatbot ===")
print(f"  Token limit     : {config['TOKEN_LIMIT']:,}")
print(f"  Summary at      : < {config['SUMMARY_THRESHOLD']} remaining")
print(f"  Closes at       : < {config['MIN_RESPONSE_TOKENS']} remaining")
print("  Commands        : 'save profile', 'history', 'tokens', 'exit'\n")


def token_status_line(in_tok: int, out_tok: int) -> str:
    return (
        f"[Tokens] This turn: {in_tok} in / {out_tok} out  |  "
        f"Total used: {tokens_used_total:,}  |  Remaining: {tokens_remaining:,}"
    )


# --------------------------------------------------------------------------- #
# 7. Interactive loop
# --------------------------------------------------------------------------- #
while True:
    # a. Hard limit check — close before doing anything else.
    if tokens_remaining < config["MIN_RESPONSE_TOKENS"]:
        print(
            f"\n> Remaining tokens ({tokens_remaining}) below minimum response "
            f"threshold ({config['MIN_RESPONSE_TOKENS']})."
        )
        close_chat()

    # b. Read user input (Ctrl-C / EOF saves and exits).
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n> Saving profile and exiting.")
        save_profile(profile, PROFILE_PATH)
        sys.exit(0)

    if not user_input:
        continue

    # c. Commands
    cmd = user_input.lower()
    if cmd in ("exit", "quit"):
        save_profile(profile, PROFILE_PATH)
        print("Goodbye!")
        sys.exit(0)
    if cmd == "save profile":
        do_save_profile()
        continue
    if cmd == "tokens":
        # Spec: re-prints the current token status line.
        print(token_status_line(last_in_tok, last_out_tok))
        continue
    if cmd == "history":
        # Spec: shows current in-session turns and message count.
        turns = len(history) // 2
        print(f"  {turns} turn(s) | {len(history)} message(s) in context:")
        for i in range(0, len(history), 2):
            print(f"  [{i // 2 + 1}] You: {history[i].content[:80]}")
        continue

    # d. Summarisation trigger — fires whenever remaining < threshold.
    if tokens_remaining < config["SUMMARY_THRESHOLD"] and history:
        print("\n> Token budget low — compressing conversation history...")
        summary_text, hist_tokens, summary_tokens = summarise_history(history)

        # Compression buys back budget: the bulky history (≈ hist_tokens worth
        # of context that was being re-sent every turn) is replaced by a small
        # summary (≈ summary_tokens). The net reclaimed budget is the difference.
        # We never let the reclaim exceed what's already been spent.
        freed = max(hist_tokens - summary_tokens, 0)
        freed = min(freed, tokens_used_total)
        tokens_used_total -= freed
        tokens_remaining = config["TOKEN_LIMIT"] - tokens_used_total

        # Step 4: replace history with exactly two messages.
        history = [
            HumanMessage(content="Previous conversation summary:"),
            AIMessage(content=summary_text),
        ]

        print("> Summary complete.")
        print(f"> Tokens freed: ~{freed:,}  |  Remaining after summary: {tokens_remaining:,}")
        print("─" * 60)

    # e. Call llm.invoke directly (NOT chain.invoke) to capture metadata.
    messages = prompt.format_messages(history=history, question=user_input)
    ai_message: AIMessage = llm.invoke(messages)

    # f. Extract content + usage_metadata.
    content = ai_message.content
    meta = ai_message.usage_metadata or {}
    in_tok = meta.get("input_tokens", 0)
    out_tok = meta.get("output_tokens", 0)

    # g. Update counters.
    tokens_used_total += in_tok + out_tok
    tokens_remaining = config["TOKEN_LIMIT"] - tokens_used_total
    last_in_tok = in_tok
    last_out_tok = out_tok

    # h. Print response + token status line.
    print(f"\nAgent: {content}\n")
    print(token_status_line(in_tok, out_tok))
    print("─" * 60)

    # i. Append to history.
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=content))