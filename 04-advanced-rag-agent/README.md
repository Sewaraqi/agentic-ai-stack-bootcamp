# Module 04 – Advanced RAG Agent

Assembles the RAG pipeline from earlier modules and wraps it in a production-grade agent with three governance layers: PII redaction at the input boundary, vector memory with contradiction detection, and an audit log.

The ReAct tool-use track that previously lived here has moved to `05-tools-and-react/`.

## Structure

```
04-advanced-rag-agent/
├── base/
│   ├── agent_base.py      # ABC: chat() + reset()
│   ├── memory_base.py     # ABC: add() + search() + clear()
│   └── retriever_base.py  # ABC: retrieve() + index()
├── agents/
│   ├── conversation_agent.py  # stateful multi-turn (from module 02)
│   └── rag_agent.py           # RAG + vector memory + PII redaction + audit log
├── services/
│   ├── llm_client.py
│   ├── embedding_service.py
│   ├── vector_memory_store.py
│   ├── document_store.py      # Pinecone + chunking + MMR
│   └── rag_pipeline.py        # retrieve → prompt → LLM with refusal threshold
├── data/corpus/               # drop .txt knowledge-base files here
├── main_01_rag_pipeline.py    # basic RAG Q&A
└── main_02_rag_agent.py       # RAG + PII redaction + contradiction detection + audit
```

## Setup

```bash
pip install -r requirements.txt
cp ../.env.example .env
# Fill in GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, and all PINECONE_* vars
```

Add your `.txt` knowledge-base files to `data/corpus/` before running either script.

## Run

```bash
python main_01_rag_pipeline.py   # basic RAG
python main_02_rag_agent.py      # full RAG agent with governance
```

## Key Concepts

### RagAgent governance (`agents/rag_agent.py`)

- **DataCards** — document every corpus file with license, PII risk, and refresh cadence
- **PII redaction** — emails and phone numbers are replaced before any downstream processing (memory, RAG, audit)
- **Contradiction detection** — LLM-as-judge compares vector memory context against the current RAG answer; flags divergence
- **Audit log** — every turn is timestamped and saved to `data/audit_log.json` on exit

### RagPipeline (`services/rag_pipeline.py`)

Retrieve → prompt → LLM with a configurable refusal threshold so the agent abstains instead of hallucinating when retrieval confidence is low. MMR retrieval can be toggled at runtime with the `mmr` command in script 01.

### Vector memory vs. document store

Two distinct Pinecone-backed (or in-process) stores:

- `vector_memory_store.py` — semantic memory over **past conversation turns** for cross-session recall
- `document_store.py` — chunked corpus for retrieval grounding
