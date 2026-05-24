"""
Assignment 02 — Few-Shot Bio Writer
Adds two example (input, output) pairs to the prompt so the model learns
the desired style from examples rather than instructions alone.
Compare output quality with assignment 01 (zero-shot) to see the difference.
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

EXAMPLE_1_INPUT = (
    "Name: Maya Goldstein\nRole: UX Designer at PixelPath Studio\nExperience: 4 years\n"
    "Skills: Figma, user research, prototyping\n"
    "Achievement: Redesigned onboarding flow, +35% retention\n"
    "Goal: Head of Design role at a product-led company\nFun fact: I illustrate a weekly webcomic"
)
EXAMPLE_1_OUTPUT = (
    "Hi! I'm Maya Goldstein, a UX Designer at PixelPath Studio with four years of experience "
    "crafting intuitive digital experiences that put users first. My core toolkit includes Figma, "
    "user research, and rapid prototyping. One of my proudest moments was redesigning our product's "
    "onboarding flow — a change that lifted user retention by 35% and became the template for all "
    "future feature launches. I'm now looking to grow into a Head of Design position at a product-led "
    "company where I can build and mentor a design team. When I'm not sketching wireframes, I'm drawing "
    "a weekly webcomic that pokes fun at the quirks of office life.\n\nTAG: maya.goldstein@school.edu"
)

EXAMPLE_2_INPUT = (
    "Name: James Okafor\nRole: Financial Analyst at Sterling Capital\nExperience: 2 years\n"
    "Skills: Excel, financial modelling, Power BI\n"
    "Achievement: Built forecasting model, +25% budget accuracy\n"
    "Goal: Corporate finance manager role\nFun fact: N/A"
)
EXAMPLE_2_OUTPUT = (
    "Hi! I'm James Okafor, a Financial Analyst at Sterling Capital with two years of experience "
    "translating complex financial data into clear, actionable insights. I specialise in Excel, "
    "financial modelling, and Power BI. Recently, I built a forecasting model that improved our "
    "annual budget accuracy by 25%, giving leadership the confidence to make faster decisions. "
    "Looking ahead, I'm eager to step into a corporate finance manager role where I can lead "
    "cross-functional planning cycles and develop the next generation of analysts.\n\nTAG: james.okafor@school.edu"
)

REAL_USER_INPUT = (
    "Name: {full_name}\nRole: {role}\nExperience: {experience}\n"
    "Skills: {skills}\nAchievement: {achievement}\nGoal: {goal}\nFun fact: {fun_fact}"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a professional bio writer. Study the example pairs and match their style exactly. "
        "Write a first-person introduction STRICTLY 120–150 words. "
        "Output ONLY the paragraph followed by a blank line and the TAG line. "
        "No headings, no preamble. If Fun fact is N/A, end naturally without mentioning it.",
    ),
    ("human", EXAMPLE_1_INPUT),
    ("ai", EXAMPLE_1_OUTPUT),
    ("human", EXAMPLE_2_INPUT),
    ("ai", EXAMPLE_2_OUTPUT),
    ("human", REAL_USER_INPUT),
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({
    "full_name": full_name, "role": role, "experience": experience,
    "skills": skills, "achievement": achievement, "goal": goal, "fun_fact": fun_fact,
})

print("\n=== Your Introduction ===\n")
print(result.strip())
print(f"\nTAG: {tag}")
