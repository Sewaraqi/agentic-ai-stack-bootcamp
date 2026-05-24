# Assignment 09 – CV Job Fit Agent

**Type:** Pair assignment  
**Based on:** `04-advanced-agents-and-tools/`

---

## Background

The tools in Assignment 07 read and extract text. The agent in Assignment 08 answers questions.
Neither one **creates** anything.

This assignment adds a third capability: **artifact generation with human approval.**

The agent reads a job description and a CV, reasons about the gap, and **proposes changes** to the user.
Only after the user confirms does the agent rewrite the CV and save the result to disk.

This pattern — **analyse → present → confirm → act** — appears in almost every production AI writing assistant.
The confirmation step is a deliberate trust boundary: the human decides whether the LLM's proposal is acceptable before any file is touched.

### Two new concepts you will implement

**Human-in-the-loop:**
The REPL pauses after the analysis phase, shows the proposals to the user, and only continues to the write phase if the user types `yes`. The agent cannot skip this gate.

**Side-effecting tools and idempotency:**
A tool that writes a file has side effects — calling it twice overwrites the first result.
This makes it `is_idempotent=False`, which tells `ToolExecutor` not to retry it automatically on failure.

---

## Step 0 — Design First

Before writing any code, produce a directory tree with every folder and file you plan to create.
For each file, write one sentence on its responsibility.
**Agree on it as a team before opening an editor.**

---

## Directory Structure

```
09-cv-job-fit-agent/
├── README.md
├── requirements.txt
├── base/
│   ├── __init__.py
│   ├── agent_base.py           # AgentBase ABC (copy from module 04)
│   └── tool_base.py            # ToolBase, ToolSchema, ToolResult (copy from module 04)
├── agents/
│   ├── __init__.py
│   └── cv_fit_agent.py         # ToolAgent configured with all three tools
├── services/
│   ├── __init__.py
│   └── llm_client.py           # LlmClient wrapper (copy from module 04)
├── tools/
│   ├── __init__.py
│   ├── jd_reader_tool.py       # Tool 1: reads a JD from data/jobs/
│   ├── cv_reader_tool.py       # Tool 2: reads a CV from data/cv/
│   └── cv_writer_tool.py       # Tool 3: saves a rewritten CV to data/cv/ (is_idempotent=False)
├── data/
│   ├── jobs/
│   │   └── sample_jd.txt       # Sample job description — backend/platform role
│   └── cv/
│       └── sample_cv.txt       # Sample CV — junior-mid developer with skill gaps
└── main_cv_fit.py              # Two-phase REPL: analyse → confirm → rewrite
```

---

## Sample Data

Two plain-text files are provided as starting fixtures. You may replace or extend them.

### `data/jobs/sample_jd.txt`
A job description for a **backend/platform engineer** role that requires skills the sample CV does not fully have:
- Docker, Kubernetes
- FastAPI (not Flask)
- CI/CD pipelines
- AWS or cloud experience
- REST API design

### `data/cv/sample_cv.txt`
A CV for a **junior–mid Python developer** with related but weaker experience:
- Flask experience (not FastAPI)
- No cloud mentions
- No CI/CD
- Basic REST experience
- Strong Python fundamentals

The **gap between the two files** is what the agent will analyse and propose to fix.

---

## What to Build

### Tool 1 — JD Reader (`tools/jd_reader_tool.py`)

**Contract:**
- Accepts `filename` (not a full path)
- Validates the file exists in `data/jobs/` and has a `.txt` extension
- Returns full file content via `ToolResult(value=...)`
- Returns `ToolResult(error=...)` on validation failure — never raises
- `is_idempotent=True`
- **Description must** clearly distinguish it from the CV reader: the LLM must never call it to read a CV

---

### Tool 2 — CV Reader (`tools/cv_reader_tool.py`)

Same validation rules as Tool 1, but scoped to `data/cv/`.

**Contract:**
- Accepts `filename` scoped to `data/cv/`
- Returns `ToolResult(error=...)` on any failure
- `is_idempotent=True`
- **Description must** make it clear this reads a candidate's CV, not the job description

---

### Tool 3 — CV Writer (`tools/cv_writer_tool.py`)

A tool that saves a rewritten CV to disk. **Has side effects.**

**Contract:**
- Accepts `filename` (output name, must end in `.txt`) and `content` (the full new CV text)
- Validates `filename` ends in `.txt` and `content` is not empty **before** writing anything
- Writes to `data/cv/<filename>`
- Returns `ToolResult(value="CV saved to data/cv/<filename>")`
- Returns `ToolResult(error=...)` on validation failure
- `is_idempotent=False` on **all** return paths — including the error path
- **Description must** instruct the LLM to wrap every tech keyword in `**double asterisks**` before calling this tool

---

### CV Fit Agent (`agents/cv_fit_agent.py`)

A `ToolAgent` configured with all three tools and a high enough `max_steps` to cover both phases
(reading two files + writing one = at least 4 steps in Phase 2 alone; budget for Phase 1 analysis on top of that).

---

### `main_cv_fit.py` — Two-Phase REPL

#### Phase 1 — Analysis

Triggered by: `fit <jd_file> <cv_file>`

The REPL calls `agent.chat()` with a prompt that instructs the agent to:
1. Read the JD using `jd_reader`
2. Read the CV using `cv_reader`
3. Return a `final_answer` containing:
   - A list of tech keywords present in the JD but missing or weak in the CV
   - Section-by-section suggestions (e.g. `"rewrite the summary to mention microservices"`)
   - A list of tech terms to highlight in the new CV

> **The prompt must explicitly tell the agent NOT to call `cv_writer` in this phase.**
> You can verify this in the trace: there must be no `[cv_writer] ACT` entry.

Print the full analysis to the terminal.

#### Confirmation Gate

After printing the analysis:
```
Apply these changes and create a new CV? (yes/no):
```
- If `no`: print `No changes made.` and return to the prompt
- If `yes`: ask for an output filename (suggest `<original>_v2.txt` as default), then proceed to Phase 2

**The file must never be written if the user types `no`.**

#### Phase 2 — Generation

Calls `agent.chat()` a second time with a prompt that includes:
- The JD filename and CV filename (so the agent re-reads both)
- The analysis from Phase 1 (passed as context in the prompt string)
- The output filename chosen by the user
- Instructions to rewrite the CV, apply the suggestions, and save using `cv_writer`

**Rules the prompt must enforce:**
- Do not invent new roles, degrees, or companies
- Add missing tech keywords only where the candidate's real experience supports it
- Improve the summary to target the specific role
- Wrap every tech keyword and tool name in `**double asterisks**`

After `agent.chat()` returns, print the agent's final message and the path to the saved file.

#### Additional Commands
```
trace   → show the Plan/Act/Observe step trace for the last agent.chat() call
exit    → quit
```

---

## Key Concepts

**Human-in-the-loop**
The REPL holds the confirmation gate — not the agent. The agent cannot bypass it because Phase 2 is never called unless the user types `yes`. This is the standard pattern for production AI writing assistants.

**`is_idempotent=False` and ToolExecutor**
`ToolExecutor` retries idempotent tools on transient failure (exponential backoff). If `cv_writer` were marked `is_idempotent=True`, a transient network error could trigger a retry and create a duplicate file or overwrite work. Marking it `False` tells the executor to call it once and surface the error immediately.

**Why `content` validation before writing?**
Once the file is written, the side effect has happened. Checking `filename` and `content` first ensures the tool fails fast before any I/O, which makes error messages actionable and prevents creating empty or mis-named files.

**Two `agent.chat()` calls, one confirmation gate**
Phase 1 and Phase 2 are separate `agent.chat()` invocations. This is intentional — it gives the human a natural pause point between analysis and action. The analysis result is passed explicitly as context in the Phase 2 prompt, not stored in agent memory.

---

## Acceptance Criteria

- [ ] `JdReaderTool` returns `ToolResult(error=...)` when file does not exist or has wrong extension
- [ ] `CvReaderTool` returns `ToolResult(error=...)` when file does not exist or has wrong extension
- [ ] `CvWriterTool` is marked `is_idempotent=False` on **all** return paths
- [ ] `CvWriterTool` rejects empty `content` and non-`.txt` filenames before writing
- [ ] Phase 1 trace contains **no** `[cv_writer] ACT` entry
- [ ] Phase 2 trace contains `[jd_reader]`, `[cv_reader]`, and `[cv_writer]` ACT entries
- [ ] The generated CV file exists on disk after a successful `fit` + `yes` confirmation
- [ ] The generated CV contains at least five `**...**`-wrapped tech keywords
- [ ] The confirmation gate cannot be bypassed — file is never written on `no`
- [ ] `max_steps` is high enough for both phases (recommend ≥ 10)

---

## Setup

```bash
pip install -r requirements.txt
cp ../../.env.example .env
# Fill in GEMINI_API_KEY
```
