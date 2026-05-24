import os
import sys
from dotenv import load_dotenv
from agents.semantic_faq_agent import SemanticFaqAgent
from services.embedding_service import EmbeddingConfig, EmbeddingService

load_dotenv()

FAQ_PAIRS = [
    ("What is RAG?", "RAG stands for Retrieval-Augmented Generation. It combines a retrieval step — fetching relevant documents from a knowledge base — with a generation step where an LLM uses those documents to produce an answer."),
    ("How does vector search work?", "Vector search converts text into high-dimensional vectors using an embedding model, then finds the nearest vectors in a database using distance metrics like cosine similarity or dot product."),
    ("What is an embedding?", "An embedding is a dense vector of floating-point numbers that represents the semantic meaning of a piece of text. Similar meanings produce vectors that are close together in vector space."),
    ("What is cosine similarity?", "Cosine similarity measures the angle between two vectors. A score of 1.0 means they point in the same direction (semantically identical), while 0.0 means they are orthogonal (unrelated)."),
    ("What is a vector database?", "A vector database stores embedding vectors alongside their source documents and provides fast approximate nearest-neighbour search. Examples include Pinecone, Weaviate, Qdrant, and ChromaDB."),
    ("When should I use RAG instead of fine-tuning?", "Use RAG when your knowledge base changes frequently or is too large to fit in a prompt. Use fine-tuning when you need the model to learn a new style or reasoning pattern that retrieval cannot express."),
    ("What chunk size should I use for RAG?", "A common starting point is 256–512 tokens with a small overlap (e.g. 50 tokens). Smaller chunks improve retrieval precision; larger chunks give the LLM more context per passage."),
    ("What is the difference between dense and sparse retrieval?", "Dense retrieval uses embedding vectors and semantic similarity. Sparse retrieval uses keyword matching with term-frequency weights like BM25. Hybrid search combines both for better coverage."),
    ("How do I evaluate a RAG pipeline?", "Key metrics include retrieval recall, answer faithfulness, and answer relevance. Frameworks like RAGAS automate these evaluations against ground-truth question-answer pairs."),
]


def main() -> None:
    embedder = EmbeddingService(
        EmbeddingConfig(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
        )
    )

    print("=== Semantic FAQ Agent ===")
    print("Loading FAQ index...")

    with SemanticFaqAgent(embedder, FAQ_PAIRS) as agent:
        print(f"Loaded {len(FAQ_PAIRS)} entries.")
        print("Commands: top <query>  |  add <question> | <answer>  |  reset  |  exit\n")

        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                sys.exit(0)

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Bye.")
                sys.exit(0)

            if user_input.lower() == "reset":
                agent.reset()
                print("Index reset to original FAQ pairs.")
                continue

            print(agent.chat(user_input))


if __name__ == "__main__":
    main()
