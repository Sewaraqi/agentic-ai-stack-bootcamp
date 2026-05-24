# Assignment 05 – Sliding Window Memory

**Concept:** Prevent unbounded context growth by limiting how many turns the LLM sees at once. When history exceeds `MAX_HISTORY_TURNS`, the oldest turn is evicted.

**What you learn:**
- Why unbounded history eventually hits the context window limit or becomes expensive
- Sliding window strategy: `history.pop(0); history.pop(0)` when `len(history) > MAX * 2`
- Trade-off: recent context vs. full history

## Files

| File | Strategy |
|------|----------|
| `main_sliding_window.py` | Sliding window — oldest turn evicted when full |

## Run

```bash
python main_sliding_window.py
```

Set `MAX_HISTORY_TURNS=2` in `.env` to see eviction happen immediately:
```
MAX_HISTORY_TURNS=2
```

## Exercise

1. Set `MAX_HISTORY_TURNS=2` and have a 5-turn conversation about different topics
2. On turn 5, ask about something from turn 1 — the model won't remember it
3. Set `MAX_HISTORY_TURNS=10` and repeat — it remembers

**Key question:** What's the right window size? It depends on:
- Average turn length (tokens per turn)
- Model's context window
- How far back users typically reference previous turns

## Why not summarize instead of drop?

Assignment 06 uses automatic summarization as an alternative — when tokens get low, the LLM compresses history into a summary instead of losing it entirely.
