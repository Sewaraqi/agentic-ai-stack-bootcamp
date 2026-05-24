"""
What is an embedding?
When you type a sentence, the computer sees a string.
'dog' and 'puppy' share no characters, yet they mean nearly the same thing.

An embedding model reads a sentence and outputs a list of ~768 numbers — a vector.
Sentences with similar meanings produce vectors that point in nearly the same direction.
An embedding turns meaning into geometry.
"""
import os
from dotenv import load_dotenv
from services.embedding_service import EmbeddingConfig, EmbeddingService

load_dotenv()

embedder = EmbeddingService(
    EmbeddingConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
    )
)

pairs = [
    ("episodic memory", "long-term memory"),
    ("vector database", "embedding index"),
    ("episodic memory", "pizza recipe"),
    ("RAG pipeline", "retrieval-augmented generation"),
    ("agent loop", "pizza recipe"),
]

print("=== 02 - Embedding Similarity Demo ===\n")
print(f"{'Text A':<32} {'Text B':<36} {'Score':>5}   Bar")
print("-" * 85)

for text_a, text_b in pairs:
    vec_a = embedder.embed(text_a)
    vec_b = embedder.embed(text_b)
    score = embedder.similarity(vec_a, vec_b)
    bar = "#" * int(score * 20)
    print(f"{text_a:<32} {text_b:<36} {score:.3f}   {bar}")

print(f"\nEach vector has {len(embedder.embed('test'))} dimensions.")
print("\nObservation:")
print("  Score near 1.0 → semantically related")
print("  Score near 0.0 → semantically unrelated")
