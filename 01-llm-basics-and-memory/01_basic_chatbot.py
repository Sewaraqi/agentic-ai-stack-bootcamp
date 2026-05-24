import os
import sys
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
    api_key=api_key,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specializing in agentic AI systems."),
    ("user", "{question}"),
])

chain = prompt | llm | StrOutputParser()

print("=== 01 - Basic Chatbot (no memory) ===")
print("Each call is independent — the LLM has no memory of previous turns.")
print("Commands: exit\n")

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
    response = chain.invoke({"question": user_input})
    print(f"\nAgent: {response}\n")
