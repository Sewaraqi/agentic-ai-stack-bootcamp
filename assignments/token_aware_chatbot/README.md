# Assignment 6 — Token-Aware Chatbot

## Concept

Token Budget Every message you send and every reply the LLM generates costs tokens.
The model has a maximum context window. Once you fill it, the model can no longer respond.
A production chatbot must track usage, warn before the limit, and shut down gracefully when the limit is hit.

Usage Metadata When you call llm.invoke(messages) directly (without StrOutputParser),
the returned AIMessage object carries a .usage_metadata attribute:
ai_message.usage_metadata.input_tokens # tokens in the prompt sent
ai_message.usage_metadata.output_tokens # tokens in the reply generated
ai_message.usage_metadata.total_tokens # input + output combined
This is how you measure exactly how much each turn costs.

#### Summarization Memory

When the token budget runs low, instead of stopping immediately,
call the LLM once more with a special prompt: "Summarize this conversation in 100 words."
Replace the full history list with just the summary.
This frees most of the used tokens and lets the conversation continue.
Hard Limit + Graceful Shutdown If even after summarization there are not enough tokens left for a response, do not call
the LLM.
Print a friendly "chat closed" message and exit. The user knows to come back later.

#### Short-Term Memory

history: list[BaseMessage] with MessagesPlaceholder — the same pattern as main_03_short_term_mem.py. Stores the current
session turns.

#### Long-Term Memory

(User Profile) A JSON file data/user_profile.json that stores personal information the user shares (name, role,
preferences). Loaded at startup and injected into the system message so the bot always knows who it is talking to — even
after the session history is cleared by summarisation.

### Goal

Build a chatbot that:

- Tracks tokens per turn and cumulatively
- Displays a token status line after every response
- Summarizes the chat automatically when the budget runs low
- Closes the session gracefully when the hard limit is reached
- Uses short-term memory (history + MessagesPlaceholder) for the current session
- Uses long-term memory (data/user_profile.json) for personal info that survives summarization

##### Token Budget Config

All three values live in the config dict and are readable from .env:

- TOKEN_LIMIT
    - Default 4000
    - Hard cap — total tokens the session may spend
- SUMMARY_THRESHOLD
    - Default 800
    - Trigger summarization when remaining < this
- MIN_RESPONSE_TOKENS
    - Default 200
    - Close chat when remaining < this

`Defaults are intentionally low so you can hit every limit quickly during testing. In production, TOKEN_LIMIT would be in the hundreds of thousands.
`

How to Count Tokens
Do not use StrOutputParser for the main conversation turns. Call llm.invoke(messages) directly — it returns an AIMessage
with metadata:

from langchain_core.messages import AIMessage

ai_message: AIMessage = llm.invoke(messages)

content: str = ai_message.content
in_tok:  int = ai_message.usage_metadata.input_tokens
out_tok: int = ai_message.usage_metadata.output_tokens

Accumulate the total yourself:
tokens_used_total += in_tok + out_tok
tokens_remaining = config[TOKEN_LIMIT] - tokens_used_total

What to Display After Every Turn
[Tokens] This turn: 312 in / 89 out | Total used: 1,203 | Remaining: 2,797

### Summarization Flow

Triggered when tokens_remaining < config[SUMMARY_THRESHOLD]:

Step 1 print:  "> Token budget low — compressing conversation history..."

Step 2 call LLM with a summarization prompt:
System: "You are a memory compressor. Be concise and factual."
Human:  "Summarize this conversation in under 100 words:\n<history as text>"

Step 3 count the tokens used by the summarization call itself

Step 4 replace history with just two messages:
HumanMessage("Previous conversation summary:")
AIMessage(<summary text>)

Step 5 print:
"> Summary complete."
"> Tokens freed: ~1,450 | Remaining after summary: 1,820"

### Hard Limit Flow

Triggered when tokens_remaining < config[MIN_RESPONSE_TOKENS]:
╔══════════════════════════════════════════╗
║ Chat session closed. ║
║ Token limit reached for today. ║
║ Your profile has been saved. ║
║ Come back in a few hours!                               ║
╚══════════════════════════════════════════╝
Do not call the LLM
Save the session profile to JSON
Exit cleanly

### Long-Term Memory: User Profile

File: data/user_profile.json

{
"name": "Sarah Cohen",
"role": "Data Analyst at FinSight Analytics",
"preferences": ["bullet points", "short answers"],
"notes": ["interested in machine learning"]
}

Loaded at startup and injected into the system message:

You are a helpful assistant.
=== USER PROFILE ===
Name: Sarah Cohen | Role: Data Analyst ...
=== END PROFILE ===

Saved when the user types save profile (prompts for name, role, and one preference)
The profile survives summarization — personal info is never lost when history is compressed

### Commands

- save profile :Prompts for name / role / preference and saves to user_profile.json
- history : Shows current in-session turns and message count
- tokens : Re-prints the current token status line
- exit / quit : Saves profile and exits 
- Ctrl-C : Saves profile and exits

##### Requirements
TOKEN_LIMIT, SUMMARY_THRESHOLD, MIN_RESPONSE_TOKENS are in the config dict, read from .env with defaults
Token counts come from ai_message.usage_metadata — not estimated
Token status line is printed after every response
Summarization is triggered automatically when tokens_remaining < SUMMARY_THRESHOLD
After summarization, history is replaced with exactly 2 messages (summary pair)
Chat is closed gracefully (no LLM call) when tokens_remaining < MIN_RESPONSE_TOKENS
data/user_profile.json is created by save profile and loaded at startup
Profile is injected into the system message so it survives history compression
Ctrl-C and exit both save the profile before exiting

Constraints
Token counts must come from ai_message.usage_metadata — do not estimate or hardcode
Do not call the LLM when remaining tokens < MIN_RESPONSE_TOKENS
Summarisation must replace history — do not keep the old history alongside the summary
data/user_profile.json must use the schema shown above (name, role, preferences, notes)
Short-term memory must use MessagesPlaceholder (same pattern as main_03_short_term_mem.py)
