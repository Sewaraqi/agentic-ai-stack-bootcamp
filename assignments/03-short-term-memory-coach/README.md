# Assignment 03 – Short-Term Memory Coach

**Concept:** Multi-turn refinement loop. The first turn generates a bio; subsequent turns refine it. `MessagesPlaceholder` keeps the full conversation context so "make it more confident" knows what it refers to.

**What you learn:**
- `MessagesPlaceholder` — why it preserves role tags vs a plain `{history}` string slot
- Multi-turn state accumulation: `history.append(HumanMessage(...)); history.append(AIMessage(...))`
- Context-dependent instructions — "make it shorter" only works if the model sees the previous output

## Run

```bash
python main.py
```

## Exercise

After the first draft, try these refinements in order:
1. "Make it sound more confident"
2. "Add a sentence about leadership"
3. "Shorten to 100 words"
4. "Revert to 120 words"

Notice how the model correctly applies each instruction relative to the current draft, not the original.
