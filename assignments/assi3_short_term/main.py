
# Short-term memory (session scratchpad) stores the conversation history in a Python list and injects all previous
# turns into every new LLM call.
# This allows multi-turn refinement — the user can ask follow-up questions and the model remembers what was already said.
import os
import sys
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

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
    ("system","You are a helpful assistant. Write a 120-150 word introduction"),
    MessagesPlaceholder(variable_name='history'), # expands to all past returns
    ("human", "{current_input}"),
])

llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model= config[GEMINI_MODEL_NAME],
    temperature=config[GEMINI_TEMPERATURE],
    api_key=config[GEMINI_API_KEY],
)

parser: StrOutputParser = StrOutputParser()
chain = prompt | llm | parser
history: list[BaseMessage] = []

print(f"\n[sending {len(history) + 1} messages]")  # system + 1 human = 1 shown

ai_response = chain.invoke({"history": history, "current_input": {
    "full_name": full_name,
    "role": role,
    "experience": experience,
    "skills": skills,
    "achievement": achievement,
    "goal": goal,
    "fun_fact": fun_fact if fun_fact else "N/A — skip this entirely",
}})
print("\n--- First Draft ---\n")
print(ai_response.strip())
print(f"\nTAG: {tag}")


# added for short-term memory: append this turn as typed message objects
history.append(HumanMessage(REAL_USER_INPUT))
history.append(AIMessage(ai_response))
latest_draft = ai_response
while True:
    try:
        user_input: str = input("Refinement (or 'done' to finish): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

    if not user_input:
        continue

    # added for agent loop: built-in exit commands
    if user_input.lower() == "done":
        print("Goodbye!")
        sys.exit(0)
    total_messages = 1 + len(history) + 1
    print(f"[sending {total_messages} messages]")


    ai_response: str = chain.invoke({"history": history, "current_input": user_input})
    print(f"\n{ai_response.strip()}")
    print(f"\nTAG: {tag}")

    # added for short-term memory: append this turn as typed message objects
    history.append(HumanMessage(user_input))
    history.append(AIMessage(ai_response))
    latest_draft = ai_response

print("\n=== Final Introduction ===\n")
print(latest_draft.strip())
print(f"\nTAG: {tag}")