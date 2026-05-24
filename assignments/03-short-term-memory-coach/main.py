"""
Assignment 03 — Short-Term Memory Coach
Generates a bio on the first turn, then allows the user to refine it
across multiple turns. Short-term memory (MessagesPlaceholder) keeps
the full conversation context so follow-up instructions land correctly.
"""
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

full_name    = input("Full name: ").strip()
role         = input("Current role / degree: ").strip()
experience   = input("Years of experience or seniority: ").strip()
skills       = input("Top 3 skills (comma-separated): ").strip()
achievement  = input("One achievement you're proud of: ").strip()
goal         = input("What you're looking for (goal): ").strip()
fun_fact_raw = input("Fun fact (optional — press Enter to skip): ").strip()
fun_fact     = fun_fact_raw if fun_fact_raw else "N/A — skip this entirely"

name_parts = full_name.lower().split()
tag = f"{name_parts[0]}.{name_parts[-1]}@school.edu" if len(name_parts) >= 2 else f"{name_parts[0]}@school.edu"

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional bio writer. Write a 120–150 word first-person introduction."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{current_input}"),
])

chain = prompt | llm | StrOutputParser()
history: list[BaseMessage] = []

first_input = (
    f"Name: {full_name}\nRole: {role}\nExperience: {experience}\n"
    f"Skills: {skills}\nAchievement: {achievement}\nGoal: {goal}\nFun fact: {fun_fact}"
)

print(f"\n[Generating first draft — {len(history) + 1} message(s) sent]")
response = chain.invoke({"history": history, "current_input": first_input})
print("\n--- First Draft ---\n")
print(response.strip())
print(f"\nTAG: {tag}\n")

history.append(HumanMessage(content=first_input))
history.append(AIMessage(content=response))
latest = response

print("You can now refine the bio. Examples:")
print("  'Make it sound more confident'")
print("  'Add more emphasis on the achievement'")
print("  'Shorten to 100 words'\n")

while True:
    try:
        user_input = input("Refinement (or 'done' to finish): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)
    if not user_input:
        continue
    if user_input.lower() == "done":
        break

    total = 1 + len(history) + 1
    print(f"[{total} message(s) sent]")
    response = chain.invoke({"history": history, "current_input": user_input})
    print(f"\n{response.strip()}\nTAG: {tag}\n")

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response))
    latest = response

print("\n=== Final Introduction ===\n")
print(latest.strip())
print(f"\nTAG: {tag}")
