import os
from dotenv import load_dotenv
from services.embedding_service import EmbeddingConfig, EmbeddingService
from services.semantic_faq import SemanticFaq, SemanticFaqConfig

load_dotenv()

FAQ_PAIRS = [
    ("What is RAG?", "RAG stands for Retrieval-Augmented Generation. It combines a retrieval step — fetching relevant documents from a knowledge base — with a generation step where an LLM uses those documents to produce an answer."),
    ("How does vector search work?", "Vector search converts text into high-dimensional vectors using an embedding model, then finds the nearest vectors in a database using distance metrics like cosine similarity or dot product."),
    ("What is an embedding?", "An embedding is a dense vector of floating-point numbers that represents the semantic meaning of a piece of text. Similar meanings produce vectors that are close together in vector space."),
    ("What is cosine similarity?", "Cosine similarity measures the angle between two vectors. A score of 1.0 means they point in the same direction (semantically identical), while 0.0 means they are orthogonal (unrelated)."),
    ("What is a vector database?", "A vector database stores embedding vectors alongside their source documents and provides fast approximate nearest-neighbour (ANN) search. Examples include Pinecone, Weaviate, Qdrant, and ChromaDB."),
    ("When should I use RAG instead of fine-tuning?", "Use RAG when your knowledge base changes frequently or is too large to fit in a prompt. Use fine-tuning when you need the model to learn a new style, format, or reasoning pattern that cannot be expressed through retrieval alone."),
    ("What chunk size should I use for RAG?", "A common starting point is 256–512 tokens with a small overlap (e.g. 50 tokens). Smaller chunks improve retrieval precision; larger chunks give the LLM more context per retrieved passage. Experiment based on your document type."),
    ("What is the difference between dense and sparse retrieval?", "Dense retrieval uses embedding vectors and semantic similarity (e.g. bi-encoder models). Sparse retrieval uses keyword matching with term-frequency weights like BM25. Hybrid search combines both for better coverage."),
    ("How do I evaluate a RAG pipeline?", "Key metrics include retrieval recall (did the right chunk get retrieved?), answer faithfulness (does the answer stay grounded in the retrieved context?), and answer relevance (does it actually address the question?). Frameworks like RAGAS automate these."),
]


def main():
    embedder = EmbeddingService(
        EmbeddingConfig(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
        )
    )
    faq = SemanticFaq(SemanticFaqConfig(), embedder)

    print("Loading FAQ index...")
    faq.load(FAQ_PAIRS)
    print(f"Loaded {len(FAQ_PAIRS)} entries.\n")
    print("Commands: 'top <query>'  |  'add <question> | <answer>'  |  'exit'")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Bye.")
            break

        if user_input.lower().startswith("top "):
            query = user_input[4:].strip()
            matches = faq.top_matches(query, n=3)
            if not matches:
                print("Index is empty.")
            for rank, (score, question, answer) in enumerate(matches, 1):
                print(f"\n  [{rank}] score={score:.3f}")
                print(f"      Q: {question}")
                print(f"      A: {answer}")
            continue

        if user_input.lower().startswith("add "):
            rest = user_input[4:]
            if " | " not in rest:
                print("Usage: add <question> | <answer>")
                continue
            question, answer = rest.split(" | ", 1)
            faq.add(question.strip(), answer.strip())
            print(f"Added: '{question.strip()}'")
            continue

        print(faq.ask(user_input))


if __name__ == "__main__":
    main()
