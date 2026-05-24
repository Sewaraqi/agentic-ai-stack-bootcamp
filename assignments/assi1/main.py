
# Purpose of the Assignment
# This assignment teaches you how to build a basic AI-powered pipeline —
# the most fundamental pattern in modern AI application development. Specifically it trains you to:

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

EXAMPLE_1_INPUT = """\
Name: Maya Goldstein
Role / Degree: UX Designer at PixelPath Studio
Experience / Seniority: 4 years
Top 3 Skills: Figma, user research, prototyping
Achievement: Redesigned the onboarding flow that increased user retention by 35%
Goal: Move into a Head of Design role at a product-led company
Fun fact: I illustrate a weekly webcomic about office life"""

# ── Example 1 expected output (first-person, 120–150 words, TAG at end) ─
EXAMPLE_1_OUTPUT = """\
Hi! I'm Maya Goldstein, a UX Designer at PixelPath Studio with four years of \
experience crafting intuitive digital experiences that put users first.
My core toolkit includes Figma, user research, and rapid prototyping. One of my \
proudest moments was redesigning our product's onboarding flow — a change that \
lifted user retention by 35% and became the template for all future feature launches.
I'm now looking to grow into a Head of Design position at a product-led company \
where I can build and mentor a design team while shaping the overall user experience \
strategy.
When I'm not sketching wireframes, I'm drawing a weekly webcomic that pokes fun at \
the quirks of office life — same eye for detail, different canvas.

TAG: maya.goldstein@school.edu"""

# ── Example 2 input block ──────────────────────────────────────────────
EXAMPLE_2_INPUT = """\
Name: James Okafor
Role / Degree: Financial Analyst at Sterling Capital
Experience / Seniority: 2 years
Top 3 Skills: Excel, financial modelling, Power BI
Achievement: Built a forecasting model that improved budget accuracy by 25%
Goal: Transition into a corporate finance manager role
Fun fact: N/A"""

# ── Example 2 expected output ──────────────────────────────────────────
# Teaches the model: no fun fact → paragraph still ends naturally,
# no awkward gap or mention of the missing fact.
EXAMPLE_2_OUTPUT = """\
Hi! I'm James Okafor, a Financial Analyst at Sterling Capital with two years of \
experience translating complex financial data into clear, actionable insights.
I specialise in Excel, financial modelling, and Power BI. Recently, I built a \
forecasting model from scratch that improved our annual budget accuracy by 25%, \
giving leadership the confidence to make faster, better-informed investment decisions.
Looking ahead, I'm eager to step into a corporate finance manager role where I can \
lead cross-functional planning cycles and develop the next generation of analysts on \
the team.
I thrive in environments that reward rigorous thinking and creative problem-solving — \
and I'm ready to bring that energy to a bigger stage.

TAG: james.okafor@school.edu"""

# ── Real user human message (named placeholders only) ─────────────────
REAL_USER_INPUT = """\
Name: {full_name}
Role / Degree: {role}
Experience / Seniority: {experience}
Top 3 Skills: {skills}
Achievement: {achievement}
Goal: {goal}
Fun fact: {fun_fact}"""

# ── Assemble the full template ─────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    # System: define the AI's role and hard constraints
    ("system",
     "You are a professional bio writer. "
     "Study the example pairs below and match their style exactly. "
     "Write a first-person introduction paragraph that is STRICTLY 120–150 words. "
     "Output ONLY the paragraph followed by a blank line and the TAG line. "
     "No headings, no preamble, no extra commentary. "
     "If Fun fact is N/A, end the paragraph naturally without mentioning it."),

    # Example pair 1 — teaches skill/achievement integration + fun fact usage
    ("human", EXAMPLE_1_INPUT),
    ("ai", EXAMPLE_1_OUTPUT),

    # Example pair 2 — teaches graceful handling of a missing fun fact
    ("human", EXAMPLE_2_INPUT),
    ("ai", EXAMPLE_2_OUTPUT),

    # Real user — all values injected via named placeholders at invoke time
    ("human", REAL_USER_INPUT),
])

llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model= config[GEMINI_MODEL_NAME],
    temperature=config[GEMINI_TEMPERATURE],
    api_key=config[GEMINI_API_KEY],
)

parser: StrOutputParser = StrOutputParser()
chain = prompt | llm | parser

introduction = chain.invoke({
    "full_name": full_name,
    "role": role,
    "experience": experience,
    "skills": skills,
    "achievement": achievement,
    "goal": goal,
    "fun_fact": fun_fact if fun_fact else "N/A — skip this entirely",
})
lines = [line for line in introduction.strip().splitlines() if not line.startswith("TAG:")]
clean_intro = "\n".join(lines).strip()

print("\n=== Your Introduction ===\n")
print(introduction.strip())
print(f"\nTAG: {tag}")