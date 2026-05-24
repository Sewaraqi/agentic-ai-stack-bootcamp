# Agentic AI Stack Bootcamp

A hands-on progression through building production-grade AI agents from scratch — from a stateless LLM call to a full RAG + tool-use agent with PII redaction and audit logging.
Aligned with the [NVIDIA NCP-AAI certification](https://www.nvidia.com/en-eu/learn/certification/agentic-ai-professional/).

**Stack:** Python · LangChain · Google Gemini · Pinecone · LangChain Google GenAI

---

## Learning Path

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| [01 – LLM Basics & Memory](./01-llm-basics-and-memory/) | Stateless chatbot → short-term → long-term memory | `ChatPromptTemplate`, `MessagesPlaceholder`, JSON persistence |
| [02 – Agent Architecture & Embeddings](./02-agent-architecture-and-embeddings/) | Modular agent design + semantic similarity | ABC pattern, `LlmClient`, `EmbeddingService`, cosine similarity |
| [03 – Vector Memory & RAG](./03-vector-memory-and-rag/) | Semantic search over history + grounded answers | `VectorMemoryStore`, Pinecone, `RagPipeline`, MMR retrieval |
| [04 – Advanced RAG Agent](./04-advanced-rag-agent/) | Production RAG with PII redaction, contradiction detection, audit log | `RagAgent`, `DataCards`, refusal threshold, `audit_log.json` |
| [05 – Tools & ReAct](./05-tools-and-react/) | Tool-use agent with ReAct loop | `ToolAgent`, `ToolExecutor`, `ToolBase`, Plan/Act/Observe cycle |
| [Assignments](./Assignments/) | Applied exercises per concept | Zero-shot, few-shot, memory patterns, token budgets, timed agents, semantic FAQ |

---

## Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com) API key (Gemini)
- A [Pinecone](https://pinecone.io) account and API key (modules 03–04 and CV agent only)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/your-username/agentic-ai-stack-bootcamp.git
cd agentic-ai-stack-bootcamp

# Copy environment variables
cp .env.example .env
# Edit .env and fill in your API keys

# Install dependencies for the module you want to run
cd 01-llm-basics-and-memory
pip install -r requirements.txt

# Run any script
python 01_basic_chatbot.py
```

Each module is self-contained with its own `requirements.txt`. Start from module 01 and follow the path — each module builds on the concepts from the previous one.

---

## Architecture Evolution

```
Module 01: user → prompt → LLM → response
Module 02: user → AgentBase → LlmClient → LLM
Module 03: user → Agent → VectorMemory + RagPipeline → LLM
Module 04: user → RagAgent → PII redaction → RagPipeline + VectorMemory → LLM → audit log
Module 05: user → ToolAgent → ToolExecutor → Tools → LLM (ReAct loop)
```
