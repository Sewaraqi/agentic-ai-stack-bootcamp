import os
import sys
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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

# MessagesPlaceholder injects the actual list of BaseMessage objects (with roles preserved).
# A plain {history} string slot would lose the Human/AI role tags the LLM needs.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specializing in agentic AI systems."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

# ---------------------------------------------------------------------------
# added for short-term memory: in-session message store (flat list of BaseMessage)
# Each turn appends HumanMessage + AIMessage — grows by 2 per turn.
# ---------------------------------------------------------------------------
history: list[BaseMessage] = []

print("=== 02 - Short-Term Memory ===")
print("Conversation history is kept in memory for the duration of this session.")
print("Commands: history, exit\n")

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
    if user_input.lower() == "history":
        if not history:
            print("  (no history yet)\n")
        else:
            for i in range(0, len(history), 2):
                print(f"  You:   {history[i].content[:100]}")
                if i + 1 < len(history):
                    print(f"  Agent: {history[i+1].content[:100]}")
            print()
        continue

    response = chain.invoke({"history": history, "question": user_input})
    print(f"\nAgent: {response}\n")

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response))
