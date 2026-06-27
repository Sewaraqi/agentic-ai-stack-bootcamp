# Module 07 – Multi-Agent Orchestration

Moves beyond a single ReAct agent to a **multi-agent system**: several specialist agents, each scoped to one domain with its own tools, coordinated by a layered set of patterns:

- **Dispatcher** — the top-level entry point. Classifies each query as single-domain or multi-domain with one LLM call, then delegates to the Router or the Orchestrator. Routing is automatic; the user never picks.
- **Router** — for single-domain queries: classifies the intent with one LLM call and dispatches to exactly one specialist.
- **Orchestrator** — for multi-domain queries: decomposes the task into subtasks, fans them out to multiple specialists, then synthesizes one cohesive answer.

Each specialist is a `ToolAgent` (the Plan → Act → Observe loop from Module 05) that the orchestration layer knows only by its **role** and **description** — never by its tools.

## Structure

```
07_multi_agents/
├── base/
│   ├── agent_base.py          # ABC: chat() + reset() + context-manager protocol
│   └── tool_base.py           # ABC: schema (ToolSchema) + run() → ToolResult
├── agents/
│   ├── tool_agent.py          # ReAct loop: Plan → Act → Observe → repeat
│   ├── specialist_agent.py    # ToolAgent + role + description (the dispatch unit)
│   ├── dispatcher_agent.py    # classifies single vs multi → Router or Orchestrator
│   ├── router_agent.py        # one LLM call → classify intent → one specialist
│   └── orchestrator_agent.py  # decompose → fan-out → synthesize
├── services/
│   ├── llm_client.py          # ChatGoogleGenerativeAI wrapper
│   ├── tool_executor.py       # registry + validation + retry + PLAN/ACT/OBSERVE traces
│   └── agent_registry.py      # maps role names → specialist instances
├── tools/
│   ├── calculator_tool.py     # add / subtract / multiply / divide
│   ├── math_rewriter_tool.py  # compound tool: LLM turns word-math into a plain expression
│   ├── unit_converter_tool.py # km↔miles, kg↔lbs, celsius↔fahrenheit
│   ├── weather_tool.py        # mock weather API with input validation
│   └── query_rewriter_tool.py # compound tool: calls the LLM to rewrite vague queries
├── main_multi_agent.py        # interactive REPL — routing is automatic
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cp ../.env.example .env
# Fill in GEMINI_API_KEY (GEMINI_MODEL_NAME and GEMINI_TEMPERATURE are optional)
```

## Run

```bash
python main_multi_agent.py
```

Just type any question — the dispatcher decides whether it needs one specialist or several.

## The four specialists

| Role            | Tools                          | Handles                                              |
|-----------------|--------------------------------|------------------------------------------------------|
| `math_agent`    | math_rewriter, calculator      | arithmetic and number computations, incl. word-math  |
| `weather_agent` | query_rewriter, weather        | weather/temperature lookups for specific cities      |
| `unit_agent`    | unit_converter                 | converting physical units (distance, weight, temp.)  |
| `general_agent` | *(none)*                       | general knowledge, definitions, conversational Q&A   |

Two specialists carry a `system_hint` that forces a tool order:

- `weather_agent` — when the user message has no explicit city, call `query_rewriter` first to extract one from the conversation context.
- `math_agent` — when the question contains words instead of digits (e.g. *"a dozen plus a score"*), call `math_rewriter` first to turn it into a plain arithmetic string, then pass that to `calculator`.

`unit_agent`'s description explicitly states it converts units and does **not** look up real-world weather — this keeps temperature-conversion queries (*"25 celsius in fahrenheit"*) from being routed to `weather_agent`.

## Commands

Anything that is not one of the commands below is treated as a query and routed automatically.

| Command                          | What it does                                              |
|----------------------------------|----------------------------------------------------------|
| `<any question>`                 | dispatcher classifies and answers (mode shown as single/multi) |
| `subtasks`                       | per-specialist results from the last multi-domain query  |
| `trace <math\|weather\|unit\|general>` | show the Plan/Act/Observe trace for a specialist   |
| `history`                        | show the current conversation history                    |
| `clear`                          | clear conversation history                               |
| `exit`                           | quit                                                     |

## Key Concepts

### Dispatcher pattern (`agents/dispatcher_agent.py`)

The single entry point for every query. One LLM call classifies the request:

- **`single`** — one specialist can fully answer → hand off to the `RouterAgent`.
- **`multi`** — the query needs results from more than one specialist → hand off to the `OrchestratorAgent`.

Unexpected LLM output falls back to `single`. `last_mode()` exposes the chosen path, printed after each answer. This removes the manual `route` / `orchestrate` distinction — the system decides.

### Router pattern (`agents/router_agent.py`)

The router makes **exactly one LLM call** — classification only. It reads role names and descriptions from the registry, picks one, and delegates. It does *not* run a ReAct loop (that is the specialist's job) and does *not* know which tools each specialist carries — so adding a specialist requires no change to the router.

`max_hops` is the **oscillation guard**: without it, a misconfigured registry could re-route indefinitely, with each dispatch looking valid in isolation while the system never produces an answer.

### Orchestrator pattern (`agents/orchestrator_agent.py`)

Three phases:

1. **Decompose** — one LLM call splits the task into a JSON array of `{role, task}` subtasks, each self-contained. When a request combines steps from different domains (e.g. a unit conversion *and* an arithmetic step), the prompt instructs the model to emit one subtask per domain rather than lumping them together.
2. **Fan-out** — subtasks run sequentially, one specialist per task, capped by `max_subtasks`.
3. **Synthesize** — one final LLM call integrates all specialist results into a single answer.

Failure modes addressed: duplicated work (shared context embedded at decompose time), inconsistent state (each specialist has its own `ToolExecutor` and trace log), and budget exhaustion (`max_subtasks`). Fan-out is sequential — independent subtasks could run in parallel as a next step.

### Agent registry (`services/agent_registry.py`)

The only coupling point between orchestration logic and specialist implementation. Role names are the message schema between agents: the dispatcher/router/orchestrator emit a role name, the registry translates it to a concrete agent. Two specialists may **share an `LlmClient`** but must **not share a `ToolExecutor`** — traces and tool registries are specialist-specific.

### Specialist agent (`agents/specialist_agent.py`)

A `ToolAgent` plus a `role` and `description`. Those two fields are the entire interface the multi-agent layer sees. Making any `ToolAgent` visible to the dispatcher/router/orchestrator is just a matter of adding them.

## Try these (in order, to see history in action)

```
What is the weather in London?
What about there?                  # vague follow-up → query_rewriter resolves "there"
What is a dozen plus a score?      # word-math → math_rewriter then calculator
Convert 100 km to miles           # → unit_agent
What is 25 degrees celsius in fahrenheit?   # → unit_agent (not weather_agent)
What is the capital of France?
Convert 5 miles to km and then multiply the result by 4   # multi → unit_agent + math_agent
```

The single-domain queries classify as **single** (Router → one specialist); the last one classifies as **multi** (Orchestrator → fan-out into a `unit_agent` and a `math_agent` subtask). After a multi-domain query, type `subtasks` to see each specialist's contribution, or `trace unit` / `trace math` to inspect the Plan/Act/Observe log.
