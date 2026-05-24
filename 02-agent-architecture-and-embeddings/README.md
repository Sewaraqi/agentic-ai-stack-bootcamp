# Module 02 – Agent Architecture & Embeddings

Introduces two foundational patterns: a clean modular agent architecture using the ABC pattern, and semantic similarity via text embeddings.

## Structure

```
02-agent-architecture-and-embeddings/
├── base/
│   └── agent_base.py          # Abstract contract all agents must satisfy
├── agents/
│   └── conversation_agent.py  # Concrete stateful multi-turn agent
├── services/
│   ├── llm_client.py          # LLM wrapper: config dataclass + chain builder
│   └── embedding_service.py   # Embedding wrapper + cosine similarity
├── main_01_modular_agent.py   # Run the conversation agent
└── main_02_embeddings.py      # Visualize semantic similarity between phrases
```

## Setup

```bash
pip install -r requirements.txt
cp ../.env.example .env
# Fill in GEMINI_API_KEY and GEMINI_EMBEDDING_MODEL
```

## Run

```bash
python main_01_modular_agent.py   # multi-turn chat with reset/history commands
python main_02_embeddings.py      # semantic similarity demo
```

## Key Design Decisions

**Why `AgentBase` as an ABC?**
The abstract base class enforces the `chat()` / `reset()` contract at instantiation time. Any subclass that skips implementing these raises `TypeError` before any code runs — a compile-time-equivalent check in Python.

**Why `LlmConfig` as a dataclass?**
Dataclasses auto-generate `__init__`, `__repr__`, and `__eq__`. All LLM configuration lives in one place, making it easy to swap models or change temperature without touching agent code.

**Why `LlmClient.build_chain()` instead of building chains in the agent?**
The agent shouldn't know how the LLM is wired. `build_chain()` returns a LangChain LCEL chain — the agent just calls `.invoke()`.

**How does cosine similarity work?**
`similarity = dot(a, b) / (|a| × |b|)` — measures the angle between two vectors, not their magnitude. Two semantically related sentences produce vectors that point in nearly the same direction, giving a score close to 1.0.
