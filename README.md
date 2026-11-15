# AI business card

A phone number that works as an interactive audio portfolio. A caller is greeted by an AI
voice agent, routed between specialist agents that run on different voice providers, and
can be transferred to a human.

Stack: Telnyx (phone), Pipecat (audio), LangGraph + LangChain (routing, state, tools).

## Status

M1 is done: the five agents, the routing, and the tools run as a text chat. No phone yet.

| Agent | Job |
|---|---|
| `classifier` | Greets, asks why the caller called, routes |
| `profile` | Answers questions about the owner from `profile.md` |
| `tour` | Explains how this system works, with live facts from the call |
| `consultant` | Asks about the caller's business and suggests uses |
| `contact` | Captures contact details, saves a lead, emails it to the owner, offers a transfer to the owner |

## Run

```sh
uv sync
cp .env.example .env        # set LLM_MODEL, the matching API key, OWNER_NAME, OWNER_PHONE
$EDITOR profile.md          # replace the placeholder facts with yours
uv run chat                 # text conversation with the agents
uv run pytest               # routing and tool tests; no API key needed
```

`LLM_MODEL` takes `<provider>:<model>`, for example `anthropic:claude-opus-5-5`,
`openai:gpt-5.5`, or `google_genai:gemini-3.8-pro`.

## Layout

```
profile.md                       the only source for claims about the owner
src/ai_call_center/
  config.py                      settings from .env
  agents/specs.py                one AgentSpec per agent: instructions, tools, handoffs, voice leg
  graph/state.py                 CallState (checkpointed) and CallContext (per call)
  graph/handoffs.py              transfer_to_<agent> tools
  graph/tools.py                 save_lead, request_owner_transfer, get_call_stats
  graph/build.py                 the agent loop; middleware swaps prompt and tools per active agent
  leads.py                       SQLite lead store
  notify.py                      lead emails to the owner over SMTP
  telephony/                     transfer seam; a recording fake for now
  chat.py                        terminal simulator
tests/                           scripted fake model, no network
```

## How routing works

Every caller turn is one graph run on the same `thread_id`. The checkpointed state holds
`active_agent`. Middleware reads it and gives the model that agent's prompt and tools. A
handoff tool changes `active_agent`; the loop calls the model again as the new agent. The
next turn goes straight to that agent.
