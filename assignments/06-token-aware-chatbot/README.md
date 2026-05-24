# Token-Aware Chatbot

A production-style chatbot that tracks token usage in real time, compresses its own conversation history when the budget runs low, and shuts down gracefully when the hard limit is reached. It combines short-term memory (the current session's turns) with long-term memory (a persistent user profile) so personal information survives even after the history is summarised away.

## Features

- **Token budgeting** — every turn's cost is read directly from the model's usage metadata, accumulated, and displayed after each reply.
- **Automatic summarisation** — when remaining tokens drop below a threshold, the bot asks the LLM to compress the conversation into a short summary and replaces the full history with it, freeing most of the used tokens.
- **Graceful shutdown** — if there aren't enough tokens left for a real response, the bot prints a friendly "session closed" message and exits without calling the LLM.
- **Short-term memory** — the current session's turns are held in a `history` list and fed back to the model via a `MessagesPlaceholder`.
- **Long-term memory** — a JSON user profile (name, role, preferences, notes) is loaded at startup and injected into the system message, so the bot always knows who it's talking to.

## Requirements

- Python 3.10+
- `langchain-core` and an LLM integration package (e.g. `langchain-openai`) that returns `usage_metadata` on `AIMessage`
- `python-dotenv` for reading configuration from `.env`

```bash
pip install langchain-core langchain-openai python-dotenv
```

## Configuration

All token-budget values live in a `config` dict and are read from `.env` with sensible defaults. The defaults are intentionally low so you can hit every limit quickly while testing — in production `TOKEN_LIMIT` would be in the hundreds of thousands.

| Config key | `.env` variable | Default | Purpose |
|---|---|---|---|
| `TOKEN_LIMIT` | `TOKEN_LIMIT` | `4000` | Hard cap — total tokens the session may spend |
| `SUMMARY_THRESHOLD` | `SUMMARY_THRESHOLD` | `800` | Trigger summarisation when remaining tokens fall below this |
| `MIN_RESPONSE_TOKENS` | `MIN_RESPONSE_TOKENS` | `200` | Close the chat when remaining tokens fall below this |

Example `.env`:

```env
TOKEN_LIMIT=4000
SUMMARY_THRESHOLD=800
MIN_RESPONSE_TOKENS=200
OPENAI_API_KEY=your-key-here
```

## Usage

```bash
python main_06_token_aware_chatbot.py
```

On the first run, if no profile exists you'll see `No user profile found — starting fresh.` followed by the startup banner:

```
=== Token-Aware Chatbot ===
  Token limit     : 4,000
  Summary at      : < 800 remaining
  Closes at       : < 200 remaining
  Commands        : 'save profile', 'history', 'tokens', 'exit'
```

After every response the bot prints a token status line:

```
[Tokens] This turn: 312 in / 89 out  |  Total used: 1,203  |  Remaining: 2,797
```

## Commands

| Command | What it does |
|---|---|
| `save profile` | Prompts for name, role, and one preference, then writes `data/user_profile.json` |
| `history` | Shows the current in-session turns and message count |
| `tokens` | Re-prints the current token status line |
| `exit` / `quit` | Saves the profile and exits |
| `Ctrl-C` | Saves the profile and exits |

## How Token Counting Works

The main conversation turns call `llm.invoke(messages)` directly — **not** through `StrOutputParser` — so the returned `AIMessage` carries usage metadata:

```python
from langchain_core.messages import AIMessage

ai_message: AIMessage = llm.invoke(messages)

content: str = ai_message.content
in_tok:  int = ai_message.usage_metadata.input_tokens
out_tok: int = ai_message.usage_metadata.output_tokens
```

The total is accumulated manually and remaining tokens derived from the limit:

```python
tokens_used_total += in_tok + out_tok
tokens_remaining   = config["TOKEN_LIMIT"] - tokens_used_total
```

Token counts are always taken from `usage_metadata` — never estimated or hardcoded.

## Summarisation Flow

Triggered when `tokens_remaining < SUMMARY_THRESHOLD`:

1. Print `> Token budget low — compressing conversation history...`
2. Call the LLM with a summarisation prompt:
   - System: *"You are a memory compressor. Be concise and factual."*
   - Human: *"Summarise this conversation in under 100 words:\n&lt;history as text&gt;"*
3. Count the tokens used by the summarisation call itself.
4. Replace `history` with exactly two messages:
   - `HumanMessage("Previous conversation summary:")`
   - `AIMessage(<summary text>)`
5. Print the result:
   ```
   > Summary complete.
   > Tokens freed: ~1,450  |  Remaining after summary: 1,820
   ```

The old history is **replaced**, not kept alongside the summary.

## Hard Limit Flow

Triggered when `tokens_remaining < MIN_RESPONSE_TOKENS`. The bot does **not** call the LLM — it saves the profile and exits cleanly:

```
╔══════════════════════════════════════════╗
║  Chat session closed.                      ║
║  Token limit reached for today.            ║
║  Your profile has been saved.             ║
║  Come back in a few hours!                 ║
╚══════════════════════════════════════════╝
```

## Long-Term Memory: User Profile

The profile is stored at `data/user_profile.json` and follows this schema:

```json
{
  "name": "Sarah Cohen",
  "role": "Data Analyst at FinSight Analytics",
  "preferences": ["bullet points", "short answers"],
  "notes": ["interested in machine learning"]
}
```

It is loaded at startup and injected into the system message so it persists across summarisation:

```
You are a helpful assistant.
=== USER PROFILE ===
Name: Sarah Cohen | Role: Data Analyst ...
=== END PROFILE ===
```

Because the profile lives in the system message rather than in `history`, compressing the conversation never loses the user's personal information.
