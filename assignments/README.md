# Assignments

Eight applied exercises, each building on a concept from the tutorial modules.

| # | Folder | Concept | Builds on |
|---|--------|---------|-----------|
| 01 | `01-zero-shot-bio-writer/` | Prompt templates, one-shot pipeline | Module 01 |
| 02 | `02-few-shot-bio-writer/` | Few-shot in-context learning | Assignment 01 |
| 03 | `03-short-term-memory-coach/` | Multi-turn refinement loop | Module 01 script 02 |
| 04 | `04-long-term-memory-coach/` | Cross-session persistence | Module 01 script 03 |
| 05 | `05-sliding-window-memory/` | Bounded history / sliding window | Assignment 04 |
| 06 | `06-token-aware-chatbot/` | Token counting + auto-summarization | Module 01 script 03 |
| 07 | `07-timed-conversation-agent/` | Decorator/wrapper pattern, performance instrumentation | Module 02 |
| 08 | `08_semantic_faq/` | In-RAM semantic search, cosine similarity router | Module 02 |

## Setup (all assignments)

```bash
pip install langchain-core langchain-google-genai python-dotenv
cp ../.env.example .env
# Fill in GEMINI_API_KEY
```
