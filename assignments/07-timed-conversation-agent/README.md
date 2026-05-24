# Assignment 07 — Timed Conversation Agent

**Based on:** `02-agent-architecture-and-embeddings`

## Overview
`ConversationAgent` works correctly but gives the operator no insight into performance. In production, slow LLM responses degrade user experience and can indicate quota throttling. Your task is to build a `TimedAgent` that tracks how long each `chat()`  call takes and exposes that data without changing the `AgentBase` contract.
## Files to create

| File | Description |
|------|-------------|
| `agents/timed_agent.py` | `TimedAgent` class |
| `main_assignment_01.py` | REPL entry point |

## Requirements

### `TimedAgent` (`agents/timed_agent.py`)

- Extends `AgentBase`; wraps an injected `LlmClient` (builds `ConversationAgent` internally)
- `chat(user_input)` — delegates to inner agent, measures elapsed time with `time.perf_counter`, records duration, returns response unchanged
- `reset()` — clears inner agent history and resets the timing list
- `stats() -> dict` — returns `{"turns": int, "avg_s": float, "min_s": float, "max_s": float}` (values rounded to 2 dp; zeros when no turns recorded)
- Usable as a context manager; `__exit__` calls `reset()` (inherited from `AgentBase`)

**Bonus:** `slow_threshold_s: float = 3.0` parameter — prints a warning when a call exceeds the threshold.

### `main_assignment_01.py`

REPL with three built-in commands:

| Command | Behaviour |
|---------|-----------|
| `stats` | Print `agent.stats()` |
| `reset` | Call `agent.reset()`, print `"Session cleared."` |
| `exit` | Exit the REPL |

## Acceptance criteria

- `isinstance(agent, AgentBase)` returns `True`
- After 3 turns, `agent.stats()["turns"] == 3`
- After `reset()`, `agent.stats()["turns"] == 0`
- `min_s <= avg_s <= max_s` always holds
- `stats` command prints all four keys
- `reset` command clears history (next turn has no memory of previous ones)
- `with` block calls `reset()` on exit even if an exception is raised
