import os
import sys
from dotenv import load_dotenv
from agents.rag_agent import DataCard, RagAgent
from services.document_store import ChunkConfig, DocumentStore, PineconeConfig
from services.embedding_service import EmbeddingService, EmbeddingConfig
from services.llm_client import LlmClient, LlmConfig
from services.rag_pipeline import RagPipeline, RagConfig
from services.vector_memory_store import VectorMemoryStore

load_dotenv(override=True)

llm = LlmClient(LlmConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
))
embedder = EmbeddingService(EmbeddingConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
))
vector_memory = VectorMemoryStore(embedder)
store = DocumentStore(
    embedder,
    PineconeConfig(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        namespace=os.getenv("PINECONE_NAMESPACE", "module_04"),
    ),
    ChunkConfig(),
    clean_on_exit=False,
)
pipeline = RagPipeline(llm, store, RagConfig(refuse_threshold=0.60))

DATA_CARDS = [
    DataCard("ncp_aai_blueprint.txt",    license="public-NVIDIA",   pii_risk="none", refresh_cadence="on exam update"),
    DataCard("rag_concepts.txt",         license="public-tutorial", pii_risk="none", refresh_cadence="manual"),
    DataCard("memory_systems.txt",       license="public-tutorial", pii_risk="none", refresh_cadence="manual"),
    DataCard("llm_safety.txt",           license="public-tutorial", pii_risk="none", refresh_cadence="manual"),
    DataCard("dirty_document_example.txt", license="example",       pii_risk="low",  refresh_cadence="n/a"),
]

agent = RagAgent(llm, vector_memory, pipeline, store, DATA_CARDS)

print("=== 02 - RAG Agent ===")
print("Data Cards:")
for card in DATA_CARDS:
    print(f"   {card.source:<40} license={card.license:<18}  pii_risk={card.pii_risk}")
print("\n=== Indexing corpus ===")
agent.index_corpus("data/corpus")
print("Done.\n")
print("Commands:")
print("  memory <query>  — search vector memory over past turns")
print("  exit            — quit and save audit log\n")
print("Tip: ask the same question twice across sessions to trigger contradiction detection.\n")

with agent:
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
        if user_input.lower().startswith("memory "):
            query = user_input[7:].strip()
            hits = vector_memory.search(query, top_k=3)
            print(f"\n[Memory search: '{query}']")
            for e in hits:
                print(f"   [{e.role}]  {e.content[:90]}")
            print()
            continue
        print(f"\nAgent: {agent.chat(user_input)}\n")
