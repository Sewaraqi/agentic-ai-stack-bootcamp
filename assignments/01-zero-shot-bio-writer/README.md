# Assignment 01 – Zero-Shot Bio Writer

**Concept:** Build a one-shot AI pipeline — collect structured CLI input, inject it into a prompt template, and generate a professional bio paragraph.

**What you learn:**
- `ChatPromptTemplate` with named `{placeholder}` slots
- LangChain LCEL pipeline: `prompt | llm | StrOutputParser()`
- Controlling output format through prompt instructions (word count, tone)
- Why the "one-shot" constraint (no loops, no memory) matters: understand the atomic LLM call before adding complexity

## Run

```bash
python main.py
```

## Exercise

Compare the output with and without the word-count constraint in the system prompt. What happens when you remove it?
