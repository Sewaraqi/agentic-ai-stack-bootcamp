# Assignment 06 – Token-Aware Chatbot

**Concept:** Track real token usage per turn via `usage_metadata` from the LLM response. When usage crosses `SUMMARY_THRESHOLD`, automatically summarize the conversation history to compress the context instead of losing it (unlike the sliding window approach).

**What you learn:**
- Reading `usage_metadata` from a raw `AIMessage` (input_tokens, output_tokens)
- Token budget management: cumulative counting across turns
- Automatic summarization as a context compression strategy
- Graceful degradation: exit cleanly when too few tokens remain

## Run

```bash
python main.py
```

## Configuration (via .env)

| Variable | Default | Effect |
|----------|---------|--------|
| `TOKEN_LIMIT` | `32000` | Total token budget for the session |
| `SUMMARY_THRESHOLD` | `0.75` | Summarize when usage hits 75% of limit |
| `MIN_RESPONSE_TOKENS` | `500` | Exit if fewer than this many tokens remain |

## Commands

| Command | Action |
|---------|--------|
| `tokens` | Show current usage and remaining budget |
| `history` | Show turns currently in context |
| `exit` | Quit |

## Sliding window vs. summarization

| Strategy | What it loses | What it keeps |
|----------|--------------|---------------|
| Sliding window (Assignment 05) | Old turns completely | Recent turns verbatim |
| Summarization (this) | Verbatim wording | Key facts and decisions |

Summarization is better for long conversations where early context matters. Sliding window is simpler and deterministic.

## How token counting works

We call `llm.invoke(messages)` directly (not through a chain) to get back an `AIMessage` object. The `usage_metadata` attribute contains `input_tokens` and `output_tokens` reported by the Gemini API.
