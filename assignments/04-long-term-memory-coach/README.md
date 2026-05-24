# Assignment 04 – Long-Term Memory Coach

**Concept:** Cross-session persistence using JSON. Past sessions are loaded at startup and injected into the system prompt so the model recalls what was discussed in previous runs.

**What you learn:**
- JSON serialization of `BaseMessage` objects
- System-prompt injection of past context at startup time
- Two-layer memory architecture:
  - **Long-term**: loaded once at startup from disk → static in system prompt
  - **Short-term**: grows per turn this session → via `MessagesPlaceholder`

## Run

```bash
python main.py
```

## Exercise

1. Run the script and chat for a few turns, then type `exit`
2. Run it again — notice it greets you with context from the first session
3. Type `recall` to inspect exactly what was loaded from disk
4. Type `history` to see only the current session's turns

## What happens when many sessions accumulate?

The system prompt grows linearly with sessions. `MAX_RECALLED_SESSIONS=3` caps this. Assignment 05 addresses the in-session growth problem with a sliding window.
