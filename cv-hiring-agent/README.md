# CV Hiring Agent — Capstone Project

> **Status:** In progress. Core architecture is complete; corpus content pending.

A production-grade multi-agent system for CV analysis and hiring decisions. Combines all patterns from modules 01–04: RAG, vector memory, PII redaction, data cards, audit logging, and multi-agent coordination.

## Architecture

```
cv-hiring-agent/
├── agents/
│   ├── cv_analyzer.py      # Analyzes uploaded CVs using LLM + vector memory
│   └── hiring_agent.py     # Conversation + RAG to support hiring decisions
├── base/
│   ├── agent_base.py
│   ├── memory_base.py
│   └── retriever_base.py
├── services/
│   ├── llm_client.py
│   ├── embedding_service.py
│   ├── document_store.py
│   ├── rag_pipeline.py
│   └── vector_memory_store.py
└── data/
    └── corpus/             # Reference documents (AI safety, RAG concepts, etc.)
```

## What it does

- **CV Analyzer Agent** — Loads and indexes candidate CVs into Pinecone. Answers semantic queries over candidate qualifications ("who has Pinecone experience?").
- **Hiring Agent** — Combines conversation + RAG to help recruiters make informed decisions. References a knowledge base on AI safety, memory systems, and RAG concepts to provide context-aware guidance.

## Governance features (inherited from Module 04)

- PII redaction before any indexing or LLM call
- Data cards per document (license, PII risk, refresh cadence)
- Contradiction detection between memory and corpus
- Full audit log saved on exit

## TODO

- [ ] Add corpus documents (contact maintainer for content)
- [ ] Implement `cv_analyzer.py` agent class
- [ ] Implement `hiring_agent.py` agent class
- [ ] Write `main.py` entry point with CLI
- [ ] Add `requirements.txt`

## Setup (once TODO items above are complete)

```bash
pip install -r requirements.txt
cp ../.env.example .env
# Fill in GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, and all PINECONE_* vars
# Add CV files and corpus documents to data/
python main.py
```
