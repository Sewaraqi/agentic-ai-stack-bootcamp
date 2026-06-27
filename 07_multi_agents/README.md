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

## The three specialists

| Role            | Tools                          | Handles                                              |
|-----------------|--------------------------------|------------------------------------------------------|
| `math_agent`    | calculator                     | arithmetic and number computations                   |
| `weather_agent` | query_rewriter, weather        | weather/temperature lookups for specific cities      |
| `general_agent` | *(none)*                       | general knowledge, definitions, conversational Q&A   |

The `weather_agent` carries a `system_hint`: when the user message has no explicit city, it must call `query_rewriter` first to extract one from the conversation context.

## Commands

Anything that is not one of the commands below is treated as a query and routed automatically.

| Command                          | What it does                                              |
|----------------------------------|----------------------------------------------------------|
| `<any question>`                 | dispatcher classifies and answers (mode shown as single/multi) |
| `subtasks`                       | per-specialist results from the last multi-domain query  |
| `trace <math\|weather\|general>` | show the Plan/Act/Observe trace for a specialist         |
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

1. **Decompose** — one LLM call splits the task into a JSON array of `{role, task}` subtasks, each self-contained.
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
What is 42 multiplied by 13?
What is the capital of France?
Compare the weather in London and Tel Aviv and calculate the temperature difference.
```

The first four classify as **single** (Router → one specialist); the last classifies as **multi** (Orchestrator → fan-out). After a multi-domain query, type `subtasks` to see each specialist's contribution, or `trace weather` to inspect the Plan/Act/Observe log.
