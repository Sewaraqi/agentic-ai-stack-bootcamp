"""
This main runs two memory systems in parallel.
ConversationAgent still has its short-term buffer --> every turn goes to the LLM unchanged.
On top of that, we write every turn into VectorMemoryStore.
When the user types search <query>, we query the vector store by meaning
instead of calling the LLM
"""
import os
import sys
from dotenv import load_dotenv
from agents.conversation_agent import ConversationAgent
from services.embedding_service import EmbeddingService, EmbeddingConfig
from services.llm_client import LlmClient, LlmConfig
from services.vector_memory_store import VectorMemoryStore

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

memory = VectorMemoryStore(embedder)
agent = ConversationAgent(llm)

print("=== 01 - Vector Memory ===")
print("Commands:")
print("  search <query>  — semantic search over past turns")
print("  exit            — quit\n")
print("Tip: chat about a few different topics, then try:")
print("     search memory types\n")

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
        if user_input.lower().startswith("search "):
            query = user_input[7:].strip()
            hits = memory.search(query, top_k=3)
            print(f"\n[Semantic search: '{query}'] — {len(hits)} result(s)")
            for i, entry in enumerate(hits, 1):
                print(f"  {i}. [{entry.role}] {entry.content[:90]}")
            print(f"\n  (total stored turns: {len(memory)})\n")
            continue

        response = agent.chat(user_input)
        memory.add("user", user_input)
        memory.add("assistant", response)
        print(f"\nAgent: {response}\n")
