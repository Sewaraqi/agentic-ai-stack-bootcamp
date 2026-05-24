"""
Assignment 01 — Zero-Shot Bio Writer
Goal: collect structured user input and generate a professional bio paragraph
using a single prompt with no examples (zero-shot).
"""
import os
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
    (
        "system",
        "You are a professional bio writer. "
        "Write a concise, engaging first-person introduction paragraph "
        "that is STRICTLY between 120 and 150 words. "
        "Output ONLY the paragraph — no headings, no preamble. "
        "If Fun fact is N/A, end the paragraph naturally without mentioning one.",
    ),
    (
        "human",
        "Write a 120–150-word introduction for:\n\n"
        "Name: {full_name}\n"
        "Role / Degree: {role}\n"
        "Experience: {experience}\n"
        "Top 3 Skills: {skills}\n"
        "Achievement: {achievement}\n"
        "Goal: {goal}\n"
        "Fun fact: {fun_fact}",
    ),
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({
    "full_name": full_name, "role": role, "experience": experience,
    "skills": skills, "achievement": achievement, "goal": goal, "fun_fact": fun_fact,
})

print("\n=== Your Introduction ===\n")
print(result.strip())
print(f"\nTAG: {tag}")
