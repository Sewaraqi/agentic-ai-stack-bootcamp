"""
Assignment 06 — Token-Aware Chatbot
Tracks real token usage per turn via usage_metadata from the LLM response.
When cumulative usage crosses SUMMARY_THRESHOLD, the LLM summarizes the
conversation history and replaces it with a compact summary — preserving
context without growing the context window indefinitely.
Exits gracefully when fewer than MIN_RESPONSE_TOKENS remain.
"""
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")

TOKEN_LIMIT = int(os.getenv("TOKEN_LIMIT", "32000"))
SUMMARY_THRESHOLD = float(os.getenv("SUMMARY_THRESHOLD", "0.75"))
MIN_RESPONSE_TOKENS = int(os.getenv("MIN_RESPONSE_TOKENS", "500"))

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
    api_key=api_key,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specializing in agentic AI systems."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

history: list[BaseMessage] = []
cumulative_tokens: int = 0
was_summarized: bool = False


def _invoke_with_token_count(messages: list[BaseMessage]) -> tuple[str, int, int]:
    """Invoke the LLM directly to capture usage_metadata alongside the response text."""
    ai_msg = llm.invoke(messages)
    meta = getattr(ai_msg, "usage_metadata", {}) or {}
    input_tokens = meta.get("input_tokens", 0)
    output_tokens = meta.get("output_tokens", 0)
    return ai_msg.content, input_tokens, output_tokens


def _summarize_history(history: list[BaseMessage]) -> str:
    """Ask the LLM to compress the current conversation into a short summary."""
    turns_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history
    )
    summary_messages = [
        SystemMessage(content="You are a concise summarizer."),
        HumanMessage(
            content=(
                f"Summarize this conversation in 150 words or fewer, "
                f"preserving all key facts and decisions:\n\n{turns_text}"
            )
        ),
    ]
    ai_msg = llm.invoke(summary_messages)
    return ai_msg.content


def tokens_remaining() -> int:
    return TOKEN_LIMIT - cumulative_tokens


print(f"\n=== 06 - Token-Aware Chatbot ===")
print(f"  Token limit      : {TOKEN_LIMIT:,}")
print(f"  Summary trigger  : {int(SUMMARY_THRESHOLD * 100)}% ({int(TOKEN_LIMIT * SUMMARY_THRESHOLD):,} tokens)")
print(f"  Min response     : {MIN_RESPONSE_TOKENS} tokens")
print("  Commands         : tokens, history, exit\n")

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
    if user_input.lower() == "tokens":
        print(
            f"  Used: {cumulative_tokens:,} / {TOKEN_LIMIT:,} tokens  "
            f"({cumulative_tokens / TOKEN_LIMIT * 100:.1f}%)"
            f"  Remaining: {tokens_remaining():,}\n"
        )
        continue
    if user_input.lower() == "history":
        if not history:
            print("  (no history yet)\n")
        else:
            print(f"  {len(history) // 2} turn(s) in context:")
            for i in range(0, len(history), 2):
                print(f"  [{i // 2 + 1}] You:   {history[i].content[:80]}")
            print()
        continue

    # Check if enough tokens remain for a meaningful response
    if tokens_remaining() < MIN_RESPONSE_TOKENS:
        print(
            f"\n[Budget exhausted — {tokens_remaining()} tokens remain, "
            f"minimum {MIN_RESPONSE_TOKENS} required. Exiting.]\n"
        )
        print("Goodbye!")
        sys.exit(0)

    # Trigger summarization when approaching the limit
    if cumulative_tokens > TOKEN_LIMIT * SUMMARY_THRESHOLD and not was_summarized:
        print(
            f"\n[Token budget at {cumulative_tokens / TOKEN_LIMIT * 100:.0f}% — "
            f"summarizing conversation history...]\n"
        )
        summary_text = _summarize_history(history)
        history.clear()
        history.append(SystemMessage(content=f"[Conversation summary]: {summary_text}"))
        was_summarized = True
        print(f"[Summary: {summary_text[:120]}...]\n")

    messages = prompt.format_messages(history=history, question=user_input)
    response, in_tok, out_tok = _invoke_with_token_count(messages)
    turn_tokens = in_tok + out_tok
    cumulative_tokens += turn_tokens

    print(f"\nAgent: {response}")
    print(f"  [turn: {in_tok} in + {out_tok} out = {turn_tokens} tokens | total: {cumulative_tokens:,}]\n")

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response))

    # Reset summarization flag after one-time use so it can trigger again if needed
    if was_summarized and len(history) > 4:
        was_summarized = False
