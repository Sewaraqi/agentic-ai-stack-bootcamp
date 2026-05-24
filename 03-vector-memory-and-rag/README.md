# Module 03 - Vector Memory & RAG

Adds a **vector memory store** (semantic recall over chat history) and a **Pinecone-backed RAG pipeline** (grounded Q&A over your own documents) on top of the agent built in module 02.

## Setup

```bash
pip install -r requirements.txt
cp ../.env.example .env
```

Fill in the following variables in `.env`:

| Variable                  | Used by                                                | Example                          |
| ------------------------- | ------------------------------------------------------ | -------------------------------- |
| `GEMINI_API_KEY`          | LLM + embeddings                                       | `AIza...`                        |
| `GEMINI_MODEL_NAME`       | LLM model (default `gemini-1.5-flash`)                 | `gemini-1.5-flash`               |
| `GEMINI_TEMPERATURE`      | LLM sampling (default `0.0`)                           | `0.0`                            |
| `GEMINI_EMBEDDING_MODEL`  | Embedding model (default `models/gemini-embedding-001`)| `models/gemini-embedding-001`    |
| `PINECONE_API_KEY`        | Vector index                                           | `pcsk_...`                       |
| `PINECONE_INDEX_NAME`     | Name of the index to use/create                        | `module-03-rag`                  |
| `PINECONE_NAMESPACE`      | Namespace inside the index (default `module_03`)       | `module_03`                      |

## Run

```bash
python main_01_vector_memory.py
python main_02_rag_pipeline.py
```

- `main_01_vector_memory.py` - Chat normally; type `search <query>` to semantically recall past turns instead of asking the LLM. Type `exit` to quit.
- `main_02_rag_pipeline.py` - On startup, indexes every `.txt` in `data/corpus/` into Pinecone. Then ask questions; the agent answers from the corpus and prints sources. Type `mmr` to toggle diverse retrieval, `exit` to quit.

## Project layout

```
03-vector-memory-and-rag/
|-- main_01_vector_memory.py     # entrypoint: chat + semantic search
|-- main_02_rag_pipeline.py      # entrypoint: index corpus + RAG Q&A
|-- agents/
|   `-- conversation_agent.py    # chat agent with short-term history
|-- base/                        # abstract interfaces
|   |-- agent_base.py
|   |-- memory_base.py
|   `-- retriever_base.py
|-- services/                    # concrete implementations
|   |-- embedding_service.py     # Gemini embeddings + cosine similarity
|   |-- llm_client.py            # Gemini chat wrapper
|   |-- vector_memory_store.py   # in-RAM semantic memory (used by main_01)
|   |-- document_store.py        # chunk + index + retrieve via Pinecone
|   `-- rag_pipeline.py          # retrieve -> ground -> generate
`-- data/corpus/                 # sample .txt files for RAG
```
