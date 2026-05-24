
# Purpose of the Assignment
# This assignment teaches you how to build a basic AI-powered pipeline — the most fundamental pattern in modern AI application development. Specifically it trains you to:
#
# Collect structured user input from the CLI and feed it dynamically into an AI model
# Write effective prompts using templates with named placeholders, rather than hardcoded strings
# Chain components together in a clean, readable pipeline (prompt → model → parser)
# Control model output through prompt instructions (word count, tone, format)
# The "one-shot" constraint (no loops, no memory) is intentional
# it forces you to understand the simplest possible AI call before moving on to more complex patterns
# like multi-turn conversations or agents.
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

GEMINI_API_KEY: str = 'gemini_api_key'
GEMINI_MODEL_NAME: str = 'gemini_model_name'
GEMINI_TEMPERATURE: str = 'gemini_temperature'

ENV_GEMINI_API_KEY: str = 'GEMINI_API_KEY'
ENV_GEMINI_MODEL_NAME: str = 'GEMINI_MODEL_NAME'
ENV_GEMINI_TEMPERATURE: str = 'GEMINI_TEMPERATURE'

config: dict[str, object] = {
    GEMINI_API_KEY: os.getenv(ENV_GEMINI_API_KEY),
    GEMINI_MODEL_NAME: os.getenv(ENV_GEMINI_MODEL_NAME, 'gemini-1.5-flash'),
    GEMINI_TEMPERATURE: float(os.getenv(ENV_GEMINI_TEMPERATURE, '0.0')),
}

if not config[GEMINI_API_KEY]:
    raise RuntimeError(f"{ENV_GEMINI_API_KEY} is not set, Copy .env.example to .env and set your key.")

llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model= config[GEMINI_MODEL_NAME],
    temperature=config[GEMINI_TEMPERATURE],
    api_key=config[GEMINI_API_KEY],
)

parser: StrOutputParser = StrOutputParser()


full_name    = input("Full name: ").strip()
role         = input("Current role / degree: ").strip()
experience   = input("Years of experience or seniority: ").strip()
skills       = input("Top 3 skills (comma-separated): ").strip()
achievement  = input("One achievement you're proud of: ").strip()
goal         = input("What you're looking for (goal): ").strip()
fun_fact_raw = input("Fun fact (optional — press Enter to skip): ").strip()
fun_fact = fun_fact_raw if fun_fact_raw else None

# Derive TAG from name: firstname.lastname@school.edu
name_parts = full_name.lower().split()
if len(name_parts) >= 2:
    tag = f"{name_parts[0]}.{name_parts[-1]}@school.edu"
else:
    tag = f"{name_parts[0]}@school.edu"

system_message = (
    "You are a professional bio writer "
    "Your task is to write a concise, engaging first-person introduction paragraph "
    "that is STRICTLY between 120 and 150 words. "
    "Do NOT include any heading, label, or preamble — output only the paragraph text itself. "
    "Write in a warm, professional tone. "
    "If no fun fact is provided, end the paragraph naturally without mentioning one."
)

human_message = (
    "Write a 120–150-word introduction paragraph for the following person:\n\n"
    "Name: {full_name}\n"
    "Role / Degree: {role}\n"
    "Experience / Seniority: {experience}\n"
    "Top 3 Skills: {skills}\n"
    "Achievement: {achievement}\n"
    "Goal: {goal}\n"
    "Fun fact: {fun_fact}\n\n"
    "Remember: output ONLY the paragraph, no extra text, strictly 120–150 words."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human",  human_message),
])

llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model= config[GEMINI_MODEL_NAME],
    temperature=config[GEMINI_TEMPERATURE],
    api_key=config[GEMINI_API_KEY],
)

parser: StrOutputParser = StrOutputParser()
chain = prompt_template | llm | parser

introduction = chain.invoke({
    "full_name": full_name,
    "role": role,
    "experience": experience,
    "skills": skills,
    "achievement": achievement,
    "goal": goal,
    "fun_fact": fun_fact if fun_fact else "N/A — skip this entirely",
})

print("\n=== Your Introduction ===\n")
print(introduction.strip())
print(f"\nTAG: {tag}")