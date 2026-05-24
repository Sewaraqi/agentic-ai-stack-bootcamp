# Assignment 07 – CV Tools

**Type:** Pair assignment  
**Based on:** `04-advanced-agents-and-tools/` (tool architecture)

---

## Background

In every tutorial so far the agent answered questions by calling the LLM directly.
Tools change that: instead of reasoning from memory alone, the LLM can now **decide to call a function**, receive the result as an observation, and continue reasoning from there.

`ToolAgent` adds a mandatory `_validate_input` step before every call.
This is the **trust boundary**: code enforces what the LLM is allowed to pass to a tool, regardless of what the LLM was prompted to do.

In this assignment you build **two CV-related tools** and verify they work independently **before** wiring them to an agent.

---

## Step 0 — Design First

Before writing any code, produce a directory tree with every folder and file you plan to create.
For each file, write one sentence on its responsibility.
**Agree on it as a team before opening an editor.**

---

## Directory Structure

```
07-cv-tools/
├── README.md
├── requirements.txt
├── base/
│   ├── __init__.py
│   └── tool_base.py            # ToolBase ABC, ToolSchema, ToolResult (copy from module 04)
├── tools/
│   ├── __init__.py
│   ├── cv_reader_tool.py       # Tool 1: reads a full CV file from data/
│   └── cv_section_extractor_tool.py  # Tool 2: extracts a named section from a CV
├── data/
│   ├── cv_alice.txt            # Sample CV — provided as test fixture
│   └── cv_bob.txt              # Sample CV — provided as test fixture
└── main_assignment_a.py        # REPL test harness — calls tools directly, no agent
```

---

## What to Build

### Tool 1 — CV Reader (`tools/cv_reader_tool.py`)

A tool that reads a plain-text CV file and returns its full content.

**Contract:**
- Accepts `filename` (not a full path — e.g. `cv_alice.txt`)
- Validates the file exists in `data/` and has a `.txt` extension
- Returns the file content as a string via `ToolResult(value=...)`
- Returns `ToolResult(error=...)` with a clear message if validation fails — never raises
- `is_idempotent=True` (reading the same file always returns the same text)

**LLM description must make it obvious:** call this tool to read an entire CV; call the section extractor to read one section.

---

### Tool 2 — CV Section Extractor (`tools/cv_section_extractor_tool.py`)

A tool that returns the text of one named section from a CV file.

**Contract:**
- Accepts input in the format `filename|section_name` (e.g. `cv_alice.txt|WORK EXPERIENCE`)
- Validates the format (must contain `|`) and that the file exists
- Searches the CV text for the section heading and returns everything between that heading and the next heading (or end of file)
- Returns a clear message if the section is not found
- `is_idempotent=True`

---

### `main_assignment_a.py` — Test Harness

A REPL that lets you test both tools **directly** — without any agent or LLM in the loop.

**Commands to support:**
```
read <filename>               → call CvReaderTool
extract <filename>|<section>  → call CvSectionExtractorTool
help                          → list commands
exit                          → quit
```

Print the raw `ToolResult` (both `.value` and `.error`) so validation and extraction behaviour is clearly visible.

**Purpose:** verify that validation and extraction work correctly *before* handing the tools to an LLM. If the harness breaks, the agent will too — fix here first.

---

### Bonus — CV Word Count Tool (`tools/cv_word_count_tool.py`)

A tool that accepts `filename` or `filename|section_name` and returns the word count.

**Rules:**
- Reuse the extraction logic from Tool 2 — do not duplicate it
- Support both whole-file and per-section counting
- Add `count <filename>` and `count <filename>|<section>` commands to the harness

---

## Key Concepts

**Why validate in the tool, not the agent?**
The LLM can be prompted to pass any string it likes. Validation inside the tool enforces the constraint at the system boundary regardless of what the LLM was told. This is the trust boundary.

**Why `ToolResult(error=...)` instead of raising an exception?**
The ToolExecutor catches exceptions and wraps them in a `ToolResult`, but a raised exception produces a generic message. Returning a structured error lets you write a clear, user-readable message that the LLM (and the harness) can act on.

**Why build a test harness before the agent?**
Testing tools in isolation is much faster than debugging them through a ReAct loop. A failing tool inside an agent produces confusing LLM output; a failing tool in the harness produces a clear error you can fix immediately.

---

## Acceptance Criteria

- [ ] `CvReaderTool` returns `ToolResult(error=...)` (not an exception) when the file does not exist
- [ ] `CvReaderTool` returns `ToolResult(error=...)` when the file is not a `.txt`
- [ ] `CvSectionExtractorTool` returns a clear message when the section heading is not found
- [ ] `CvSectionExtractorTool` returns `ToolResult(error=...)` when the input format is wrong (no `|`)
- [ ] The harness prints both `.value` and `.error` for every tool call
- [ ] Both tools pass the harness tests with `cv_alice.txt` and `cv_bob.txt`
- [ ] (Bonus) Word count tool reuses section extraction — no copy-pasted logic

---

## Setup

```bash
pip install -r requirements.txt
cp ../../.env.example .env
# GEMINI_API_KEY only needed for the bonus word count if you add LLM summarization
```
