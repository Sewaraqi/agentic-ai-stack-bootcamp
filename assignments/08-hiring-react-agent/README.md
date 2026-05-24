# Assignment 08 – Hiring ReAct Agent

**Type:** Pair assignment  
**Based on:** `04-advanced-agents-and-tools/` + your tools from Assignment 07

---

## Background

Assignment 07 gave you individual tools and a test harness.
Now you hand those tools to an LLM and **let it decide** which ones to call — and in what order — to answer a question it cannot answer from memory alone.

This is the **ReAct loop** (Reason + Act):

```
Thought     → the LLM reasons about what to do next
Action      → the LLM names a tool and provides its input
Observation → the tool runs and returns a result
Thought     → the LLM incorporates the result and reasons again
...
Final Answer → the LLM concludes
```

`ToolAgent` runs this loop. It parses the LLM output, dispatches to the correct tool by name, injects the observation, and enforces a hard stop (`max_steps`) so the loop cannot run forever.

In this assignment the agent's job is to answer **hiring questions** by reading CVs and a job description through its tools.

---

## Step 0 — Design First

Before writing any code, produce a directory tree with every folder and file you plan to create.
For each file, write one sentence on its responsibility.
**Agree on it as a team before opening an editor.**

---

## Directory Structure

```
08-hiring-react-agent/
├── README.md
├── requirements.txt
├── base/
│   ├── __init__.py
│   ├── agent_base.py           # AgentBase ABC (copy from module 04)
│   └── tool_base.py            # ToolBase, ToolSchema, ToolResult (copy from module 04)
├── agents/
│   ├── __init__.py
│   └── hiring_agent.py         # ReAct agent: receives tools, runs ToolAgent loop
├── services/
│   ├── __init__.py
│   └── llm_client.py           # LlmClient wrapper (copy from module 04)
├── tools/
│   ├── __init__.py
│   ├── cv_reader_tool.py       # From Assignment 07 (copy or import)
│   ├── cv_section_extractor_tool.py  # From Assignment 07 (copy or import)
│   └── job_match_scorer_tool.py      # New: compound tool — file I/O + LLM reasoning
├── data/
│   ├── cv_alice.txt            # Sample CV — candidate A
│   ├── cv_bob.txt              # Sample CV — candidate B
│   └── jd_backend.txt          # Sample job description
└── main_assignment_b.py        # REPL: loads agent with all tools, prints ReAct trace
```

---

## What to Build

### New Tool — Job Match Scorer (`tools/job_match_scorer_tool.py`)

A **compound tool**: it does file I/O AND calls the LLM internally to reason about fit.

**Contract:**
- Accepts `cv_file|jd_file` (e.g. `cv_alice.txt|jd_backend.txt`)
- Validates the format and that both files exist in `data/`
- Reads both files
- Calls the LLM with both texts and a focused prompt asking for a 3–5 sentence fit assessment
- Returns `ToolResult(value=<assessment text>)`
- Returns `ToolResult(error=...)` on any validation failure
- `is_idempotent=True` (same files → same assessment, deterministically)

> **Note:** This tool calls the LLM internally. The outer agent calls it as a single action and uses the returned assessment as an observation in its own reasoning chain.

---

### Hiring Agent (`agents/hiring_agent.py`)

An agent that:
- Receives all tools at construction time
- Uses `ToolAgent` (from module 04) with a ReAct system prompt
- Enforces a `max_steps` budget (at least 8 — reading two CVs + scoring + comparing = many steps)
- Maintains `chat_history` so follow-up questions have context

The agent should be able to answer:
- `"Who is better suited for this backend role — Alice or Bob?"`
- `"What is Alice's biggest weakness for this position?"`
- `"List the required skills Bob is missing."`

**The LLM decides which tools to call. You do not hard-code the sequence.**

---

### `main_assignment_b.py` — REPL

A REPL that:
- Loads `cv_alice.txt`, `cv_bob.txt`, and `jd_backend.txt` at startup (prints confirmation)
- Creates the agent with all tools registered
- Lets the user ask any hiring-related question in natural language
- Prints the full **ReAct trace** (Thought / Action / Observation) after each answer

**Commands to support:**
```
tools   → print name + description of every registered tool
trace   → re-print the Plan/Act/Observe trace for the last query
exit    → quit
```

---

### Bonus — Bias Check Tool (`tools/bias_check_tool.py`)

A tool that takes a plain-text job match assessment and asks the LLM to identify language that could reflect unconscious bias (gender, age, nationality assumptions).

**Rules:**
- Input is the assessment text directly (not a filename)
- Returns a list of flagged phrases and the reason each was flagged
- Returns `ToolResult(value="No bias indicators found.")` if clean
- Wire it into the agent so it can be called after a match assessment is produced

---

## Key Concepts

**Why does the agent not hard-code the tool sequence?**
The LLM reads the tool descriptions and reasons about which one to call. This makes the agent generalise: you can add tools, change the question, or change the data without modifying the agent code.

**What is a compound tool?**
A tool that performs I/O and calls the LLM internally. The outer agent treats it as a black box — it sees one action and one observation. This is useful when a sub-task (fit scoring) is always done the same way and does not need to be split into separate ReAct steps.

**Why `max_steps` matters?**
Without a hard stop, a confused or looping LLM can produce hundreds of tool calls and a very large bill. `max_steps` is a safety valve — set it high enough for your expected workflow but not unlimited.

**Why print the trace?**
The trace shows you exactly what the LLM was thinking and which tools it chose. This is the primary debugging tool for ReAct agents — if the agent gives a wrong answer, the trace shows where its reasoning diverged.

---

## Acceptance Criteria

- [ ] `JobMatchScorerTool` validates both filenames before reading anything
- [ ] The agent answers "who fits better" using `job_match_scorer_tool` — visible in trace
- [ ] The agent answers section-specific questions using `cv_section_extractor_tool` — visible in trace
- [ ] `max_steps` is high enough for a two-candidate comparison (at least 8)
- [ ] Follow-up questions reference prior context (chat history is maintained)
- [ ] `tools` command lists all registered tools with descriptions
- [ ] (Bonus) `bias_check_tool` flags gendered language in a test assessment

---

## Setup

```bash
pip install -r requirements.txt
cp ../../.env.example .env
# Fill in GEMINI_API_KEY
```
