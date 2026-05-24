# Module 03 – Vector Memory & RAG

Two scripts: semantic search over conversation history, and a full Retrieval-Augmented Generation (RAG) pipeline backed by Pinecone.

## Structure

```
03-vector-memory-and-rag/
├── base/
│   ├── agent_base.py        # ABC: chat() + reset()
│   ├── memory_base.py       # ABC: add() + search() + clear(); MemoryEntry dataclass
│   └── retriever_base.py    # ABC: retrieve() + index(); RetrievalResult dataclass
├── agents/
│   └── conversation_agent.py
├── services/
│   ├── llm_client.py
│   ├── embedding_service.py
│   ├── vector_memory_store.py  # In-memory semantic store (cosine similarity)
│   ├── document_store.py       # Pinecone-backed store with chunking + MMR
│   └── rag_pipeline.py         # Retrieve → prompt → LLM; refusal threshold
├── data/corpus/                # Drop .txt files here before running script 02
├── main_01_vector_memory.py
└── main_02_rag_pipeline.py
```

## Setup

```bash
pip install -r requirements.txt
cp ../.env.example .env
# Fill in GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, and all PINECONE_* vars
```

Add your `.txt` knowledge base files to `data/corpus/` before running script 02.

## Run

```bash
python main_01_vector_memory.py   # semantic memory search over chat history
python main_02_rag_pipeline.py    # grounded Q&A over your document corpus
```

## Key Concepts

**VectorMemoryStore vs plain history list**
A flat list returns the N *most recent* turns. A vector store returns the N *most semantically relevant* turns — useful when the conversation spans many topics.

**Why Pinecone instead of in-memory?**
In-memory embeddings don't survive restarts and don't scale. Pinecone persists the index so re-indexing isn't required every session (set `clean_on_exit=False`).

**Refusal threshold**
`RagPipeline` refuses to answer when the best retrieved chunk scores below `refuse_threshold` (default 0.30 cosine similarity). This prevents hallucination when the corpus doesn't contain the answer.

**MMR — Maximal Marginal Relevance**
Standard similarity search returns the N most similar chunks, which are often near-duplicates. MMR balances relevance and diversity — it picks chunks that are relevant but also different from each other, giving the LLM broader context.

**Chunking strategy**
Documents are split with `RecursiveCharacterTextSplitter` at `chunk_size=400, chunk_overlap=40`. Overlap ensures a sentence spanning a chunk boundary is captured in both chunks so it's never missed during retrieval.
