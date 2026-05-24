# Assignment 02 — Semantic FAQ

Based on: `tutorial_02/`  
New file you will create: `services/semantic_faq.py`  
Entry point you will create: `main_assignment_02.py`

## Background

`EmbeddingService.similarity()` computes cosine similarity between two vectors. In `tutorial_02` you used it to compare sentence pairs. Now you will use it to build a `SemanticFAQ` — a lookup table that finds the closest pre-written question to any user query and returns the corresponding answer.

This is the simplest possible form of semantic search: no vector database, no external API call at query time — just cosine distance against a small in-RAM index of embedded FAQ questions.

## What to build

### `SemanticFaqConfig` dataclass in `services/semantic_faq.py`

```python
threshold: float = 0.70    # minimum cosine score to return a match
```

### `SemanticFaq` class in `services/semantic_faq.py`

Takes an `EmbeddingService` and a `SemanticFaqConfig` on construction.

**Required methods:**

**`load(pairs: list[tuple[str, str]]) -> None`**  
Accepts a list of `(question, answer)` tuples. Embeds every question and stores the vectors in a private list alongside the original text pairs. Calling `load()` a second time replaces the previous index entirely.

**`ask(query: str) -> str`**  
Embeds the query, computes cosine similarity against every stored question vector, finds the best match.
- If the best score >= `threshold`: return the corresponding answer string.
- If below threshold: return `"No matching FAQ entry found. (best score: X.XX)"` where `X.XX` is the best score rounded to 2 decimal places.

**`top_matches(query: str, n: int = 3) -> list[tuple[float, str, str]]`**  
Returns the top-n matches as a list of `(score, question, answer)` tuples, sorted by score descending. Used for debugging and demo purposes. Must work even when `n > len(stored_pairs)`.

### `main_assignment_02.py`

A REPL that:
- Builds a `SemanticFaq` loaded with at least 8 FAQ pairs on the topic of **RAG, embeddings, and vector search** (write these yourself — they do not need to be factually perfect, just varied enough to test the router).
- Supports the following commands:

| Command | Behaviour |
|---------|-----------|
| `top <query>` | Print the top-3 matches with scores |
| `exit` | Exit the REPL |
| `<any other input>` | Call `faq.ask(input)` and print the result |

## Acceptance criteria

- `load()` stores exactly as many entries as the list it received.
- After `load()`, querying with a question that is semantically identical to a stored one returns the correct answer.
- Querying with a clearly off-topic question (e.g. `"what is the capital of France?"`) returns the `"No matching FAQ"` message.
- `top_matches()` always returns at most `n` items and they are sorted highest-score first.
- Calling `load()` a second time with a different list replaces the index — queries against the old questions no longer return old answers.
- `ask()` never calls the LLM — it uses only `EmbeddingService.embed()` and `EmbeddingService.similarity()`.

## Hints

- Store vectors alongside their source text: `list[tuple[list[float], str, str]]` where the tuple is `(vector, question, answer)`.
- `embed()` is called once per question at `load()` time, once per query at `ask()` time — not multiple times.
- `top_matches()` can reuse the same cosine loop as `ask()` — consider extracting a private `_score_all(query_vector)` helper.
- The `gemini-embedding-001` baseline for semantically unrelated text is ~0.55, so `threshold=0.70` is a reasonable starting point. If your FAQ questions are very broad, lower it; if they are very specific, raise it.

## Bonus

Add a `SemanticFaq.add(question: str, answer: str) -> None` method that appends a single new FAQ entry without reloading the whole index. Then add an `add <question> | <answer>` command to the REPL that calls it. The pipe `|` separates question from answer in the command string.

## Structure

```
08_semantic_faq/
├── base/
│   └── agent_base.py          # Abstract contract all agents must satisfy
├── agents/
│   └── conversation_agent.py  # Concrete stateful multi-turn agent
├── services/
│   ├── llm_client.py          # LLM wrapper: config dataclass + chain builder
│   ├── embedding_service.py   # Embedding wrapper + cosine similarity
│   └── semantic_faq.py        # SemanticFaqConfig + SemanticFaq  ← you build this
├── main.py                    # Tutorial entry point
└── main_assignment_02.py      # Assignment REPL  ← you build this
```

## Setup

```bash
pip install -r requirements.txt
cp ../.env.example .env
# Fill in GEMINI_API_KEY and GEMINI_EMBEDDING_MODEL
```

## Run

```bash
python main.py                  # tutorial modular agent
python main_assignment_02.py    # assignment REPL
```
