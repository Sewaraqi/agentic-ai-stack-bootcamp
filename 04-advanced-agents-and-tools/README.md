# Module 04 – Advanced Agents & Tools

The capstone module. Assembles every component built across modules 01–03 and adds three new production-grade layers: the ReAct tool-use loop, PII redaction at the input boundary, and an audit log.

## Structure

```
04-advanced-agents-and-tools/
├── base/
│   ├── agent_base.py      # ABC: chat() + reset()
│   ├── memory_base.py     # ABC: add() + search() + clear()
│   ├── retriever_base.py  # ABC: retrieve() + index()
│   └── tool_base.py       # ABC: schema (ToolSchema) + run() → ToolResult
├── agents/
│   ├── conversation_agent.py  # stateful multi-turn (from module 02)
│   ├── rag_agent.py           # RAG + vector memory + PII redaction + audit log
│   └── tool_agent.py          # ReAct loop: Plan → Act (tool call) → Observe → repeat
├── services/
│   ├── llm_client.py
│   ├── embedding_service.py
│   ├── vector_memory_store.py
│   ├── document_store.py      # Pinecone + chunking + MMR
│   ├── rag_pipeline.py        # retrieve → prompt → LLM with refusal threshold
│   └── tool_executor.py       # registry + validation + retry + PLAN/ACT/OBSERVE traces
├── tools/
│   ├── calculator_tool.py     # add / subtract / multiply / divide
│   └── weather_tool.py        # mock weather API with input validation
├── data/corpus/               # drop .txt knowledge-base files here
├── main_01_rag_pipeline.py    # basic RAG Q&A
├── main_02_rag_agent.py       # RAG + PII redaction + contradiction detection + audit
└── main_03_tool_agent.py      # ReAct loop with calculator and weather tools
```

## Setup

```bash
pip install -r requirements.txt
cp ../.env.example .env
# Fill in GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, and all PINECONE_* vars
```

Add your `.txt` knowledge-base files to `data/corpus/` before running scripts 01 or 02.

## Run

```bash
python main_01_rag_pipeline.py   # basic RAG
python main_02_rag_agent.py      # full RAG agent with governance
python main_03_tool_agent.py     # ReAct tool agent
```

## Key Concepts

### ReAct Loop (tool_agent.py)
The agent follows a strict Plan / Act / Observe cycle:
1. **Plan** — LLM sees the system prompt (tool schemas) and decides what to do next
2. **Act** — `ToolExecutor` validates and runs the chosen tool
3. **Observe** — result is injected back as a message, LLM decides next step
4. Repeat until `final_answer` action or `max_steps` is hit

The LLM communicates exclusively via JSON (`{"action": "...", "args": {...}}`). `_parse_json()` strips markdown fences since LLMs often wrap JSON in them despite instructions.

### ToolExecutor (services/tool_executor.py)
Three responsibilities:
- **Registry** — maps tool schema names to instances; rejects invented names immediately
- **Validation** — checks required args are present before running
- **Retry** — exponential backoff for idempotent tools; skips retry for tools with side effects (`is_idempotent=False`)

### RagAgent governance (agents/rag_agent.py)
- **DataCards** — document every corpus file with license, PII risk, and refresh cadence
- **PII redaction** — emails and phone numbers are replaced before any downstream processing (memory, RAG, audit)
- **Contradiction detection** — LLM-as-judge compares vector memory context against the current RAG answer; flags divergence
- **Audit log** — every turn is timestamped and saved to `data/audit_log.json` on exit

### Why `is_idempotent` matters
A calculator call with the same args always returns the same result — safe to retry. A weather API call has no side effects but a real payment API would. Setting `is_idempotent=False` prevents duplicate actions on transient failures.
