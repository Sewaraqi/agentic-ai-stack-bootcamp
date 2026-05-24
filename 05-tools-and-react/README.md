# Module 05 – Tools & ReAct

Introduces tool-use: the agent decides at each step whether to call a tool, read the tool's result, and either call another tool or produce a final answer. The pattern is the classic **ReAct** loop (Plan → Act → Observe), with a `ToolExecutor` enforcing the boundary between the LLM's free-form decisions and the actual tool calls.

## Structure

```
05-tools-and-react/
├── base/
│   ├── agent_base.py      # ABC: chat() + reset()
│   └── tool_base.py       # ABC: schema (ToolSchema) + run() → ToolResult
├── agents/
│   └── tool_agent.py      # ReAct loop: Plan → Act (tool call) → Observe → repeat
├── services/
│   ├── llm_client.py
│   └── tool_executor.py   # registry + validation + retry + PLAN/ACT/OBSERVE traces
├── tools/
│   ├── calculator_tool.py # add / subtract / multiply / divide
│   └── weather_tool.py    # mock weather API with input validation
└── main_01_tool_agent.py  # ReAct loop with calculator and weather tools
```

## Setup

```bash
pip install -r requirements.txt
cp ../.env.example .env
# Fill in GEMINI_API_KEY (Pinecone vars are NOT required for this module)
```

## Run

```bash
python main_01_tool_agent.py
```

## Key Concepts

### ReAct Loop (`agents/tool_agent.py`)

The agent follows a strict Plan / Act / Observe cycle:

1. **Plan** — LLM sees the system prompt (tool schemas) and decides what to do next
2. **Act** — `ToolExecutor` validates and runs the chosen tool
3. **Observe** — result is injected back as a message, LLM decides next step
4. Repeat until `final_answer` action or `max_steps` is hit

The LLM communicates exclusively via JSON (`{"action": "...", "args": {...}}`). `_parse_json()` strips markdown fences since LLMs often wrap JSON in them despite instructions.

### ToolExecutor (`services/tool_executor.py`)

Three responsibilities:

- **Registry** — maps tool schema names to instances; rejects invented names immediately
- **Validation** — checks required args are present before running
- **Retry** — exponential backoff for idempotent tools; skips retry for tools with side effects (`is_idempotent=False`)

Every Plan/Act/Observe event is appended to an internal trace list so you can reconstruct exactly what happened (`trace` command at the prompt).

### Why `is_idempotent` matters

A calculator call with the same args always returns the same result — safe to retry. A weather API call has no side effects but a real payment API would. Setting `is_idempotent=False` prevents duplicate actions on transient failures.

## Try these queries

- `What is 42 multiplied by 13?`
- `What is the weather in Tel Aviv?`
- `What is (15 + 7) multiplied by 2?` — two tool calls
- `What is the weather on the moon?` — error: unknown city
- `What is 10 divided by 0?` — error: business rule

Type `trace` to inspect the Plan/Act/Observe log from the last query.
