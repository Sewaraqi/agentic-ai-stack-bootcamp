# Module 01 – LLM Basics & Memory

Three progressive scripts showing how to evolve a stateless chatbot into one with full memory.

## Scripts

| Script | What it adds | Key concept |
|--------|-------------|-------------|
| `01_basic_chatbot.py` | Baseline — no memory | `ChatPromptTemplate` → LLM → `StrOutputParser` |
| `02_short_term_memory.py` | In-session history | `MessagesPlaceholder`, `HumanMessage`, `AIMessage` |
| `03_long_term_memory.py` | Cross-session persistence | JSON file store, system-prompt injection at startup |

## Setup

```bash
pip install -r requirements.txt
cp ../.env.example .env
# Fill in GEMINI_API_KEY in .env
```

## Run

```bash
python 01_basic_chatbot.py
python 02_short_term_memory.py
python 03_long_term_memory.py
```

### Commands available in script 03
| Command | Action |
|---------|--------|
| `history` | Print all turns from this session |
| `recall` | Print past sessions loaded from disk |
| `save` | Force-save current session to disk mid-conversation |
| `exit` / `quit` | Save and exit |

## Concepts

**Why `MessagesPlaceholder` instead of `{history}` as a string?**
A plain string slot would inject history as raw text, losing role information (`HumanMessage` vs `AIMessage`). `MessagesPlaceholder` injects the actual typed message objects so the LLM sees properly structured role-tagged turns.

**What happens when history gets very long?**
The LLM call fails or silently truncates once the context window is exceeded. Module 01 assignment 05 addresses this with a sliding-window strategy.

**Long-term memory strategy used here:**
Past sessions are serialized to JSON and loaded once at startup, injected into the system prompt as a text block. The current session's history is handled separately via `MessagesPlaceholder`.
