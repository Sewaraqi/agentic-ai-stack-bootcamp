import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from services.document_store import PineconeConfig, DocumentStore, ChunkConfig
from services.embedding_service import EmbeddingService, EmbeddingConfig
from services.llm_client import LlmClient, LlmConfig
from services.rag_pipeline import RagConfig, RagPipeline

load_dotenv()

llm = LlmClient(LlmConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
))
embedder = EmbeddingService(EmbeddingConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
))
store = DocumentStore(
    embedder,
    PineconeConfig(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        namespace=os.getenv("PINECONE_NAMESPACE", "module_03"),
    ),
    ChunkConfig(chunk_size=400, chunk_overlap=40),
)
rag = RagPipeline(llm, store, RagConfig(refuse_threshold=0.60))

CORPUS = "data/corpus"
print(f"=== Indexing corpus: {CORPUS} ===")
docs = []
for path in Path(CORPUS).glob("*.txt"):
    file_docs = store.load_file(str(path))
    docs.extend(file_docs)
    print(f"   {path.name:<40}  {len(file_docs)} chunk(s)")
store.index(docs)
print(f"   Total: {len(docs)} chunks indexed\n")

print("=== 02 - RAG Pipeline ===")
print("Commands:")
print("  mmr   — toggle MMR retrieval (diverse results)")
print("  exit  — quit\n")
print("Try these queries:")
print("  'What are the NCP-AAI blueprint weights?'")
print("  'What is chunk overlap and why does it matter?'")
print("  'What is the best pizza recipe?'  ← should refuse\n")

with store:
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            sys.exit(0)
        if user_input.lower() == "mmr":
            rag.config.use_mmr = not rag.config.use_mmr
            status = "ON — diverse results" if rag.config.use_mmr else "OFF — similarity only"
            print(f"MMR retrieval: {status}\n")
            continue

        answer, results = rag.answer_with_sources(user_input)
        print(f"\nAgent: {answer}")
        if results:
            seen: set[str] = set()
            for r in results:
                src = r.document.metadata.get("source", "?")
                if src not in seen:
                    score_str = f"score={r.score:.2f}" if r.score < 1.0 else "MMR"
                    print(f"  > {src}  ({score_str})")
                    seen.add(src)
        print()
