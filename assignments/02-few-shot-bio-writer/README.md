# Assignment 02 – Few-Shot Bio Writer

**Concept:** Improve prompt quality by embedding (human, ai) example pairs. The model learns the desired tone and structure from examples rather than instructions alone.

**What you learn:**
- Few-shot prompting: embedding `("human", input)` / `("ai", output)` pairs in `ChatPromptTemplate`
- How examples teach format, tone, and edge-case handling (e.g. missing fun fact)
- In-context learning — no fine-tuning required

## Run

```bash
python main.py
```

## Exercise

Run assignment 01 and assignment 02 with the same input. Compare:
- Word count precision
- Tone consistency
- Handling of the missing fun-fact case

Which produces better output? Why?

## Key insight

The second example (`EXAMPLE_2_INPUT`) teaches the model how to handle `Fun fact: N/A` — end the paragraph naturally without an awkward gap. Zero-shot prompting rarely handles edge cases this gracefully.
