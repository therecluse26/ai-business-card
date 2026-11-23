"""Agent specs: provider-neutral descriptions of each voice agent.

One spec runs on a voice leg (phone) or on a text model (tests, chat.py).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# Voice legs. The id is a setting; the label is what the tour agent says aloud.
LEGS: dict[str, str] = {
    "openai_realtime": "OpenAI gpt-realtime-2.1-mini, a speech-to-speech model",
    "gemini_live": "Google gemini-3.8-live, a speech-to-speech model",
    "nova_sonic": "Amazon Nova 2.5 Sonic, a speech-to-speech model",
}

# Short platform names, for the introduction line an agent says after a handoff.
LEG_PLATFORMS: dict[str, str] = {
    "openai_realtime": "OpenAI",
    "gemini_live": "Google Gemini",
    "nova_sonic": "Amazon Nova",
}

DEFAULT_LEGS: dict[str, str] = {
    "classifier": "openai_realtime",
    "profile": "gemini_live",
    "tour": "openai_realtime",
    "consultant": "gemini_live",
    "contact": "openai_realtime",
    "feedback": "openai_realtime",
}

SPECIALISTS: tuple[str, ...] = ("profile", "tour", "consultant", "contact", "feedback")

# One voice per agent, so a caller hears the handoff. Genders alternate along the common handoff
# paths: the classifier (female) hands to male voices; contact (female) is reached from male ones.
# Voice names belong to a provider, so each leg has its own map with the same genders.
LEG_VOICES: dict[str, dict[str, str]] = {
    "openai_realtime": {
        "classifier": "marin",  # female
        "profile": "cedar",  # male
        "tour": "ash",  # male
        "consultant": "ballad",  # male
        "contact": "sage",  # female, calm. coral sounded sharp to the owner on 2026-11-16.
        "feedback": "shimmer",  # female. Of the voices no agent uses here, the one with a measured level.
    },
    "gemini_live": {
        "classifier": "Aoede",  # female
        "profile": "Charon",  # male, deep
        "tour": "Orus",  # male
        "consultant": "Puck",  # male, bright
        "contact": "Kore",  # female
        "feedback": "Leda",  # female
    },
    # Nova has few English voices, so agents share them. In normal use one agent runs on Nova.
    "nova_sonic": {
        "classifier": "tiffany",  # female, US
        "profile": "matthew",  # male, US
        "tour": "matthew",
        "consultant": "matthew",
        "contact": "amy",  # female, British
        "feedback": "amy",  # a caller reaches it mostly from the classifier, which is tiffany
    },
}
DEFAULT_VOICES: dict[str, str] = LEG_VOICES["openai_realtime"]

# Speech level of each voice in dBFS, measured on the phone side (8 kHz PCMU) on 2026-11-15.
# The voice leveler starts from these, so a voice is at the target level from its first word.
# A voice that is not listed is measured during the call.
VOICE_LEVELS_DBFS: dict[str, float] = {
    "marin": -21.7,
    "sage": -33.0,  # measured 2026-11-16
    "shimmer": -19.1,  # measured 2026-11-16, when the contact agent got a new voice
    "ash": -24.5,
    "Charon": -15.0,
    "Puck": -20.4,
}


def parse_agent_legs(text: str) -> dict[str, str]:
    """ "consultant=nova_sonic, tour=gemini_live" as {agent: leg}. An unknown agent or leg is an error,
    so a typing mistake in `.env` stops the server at start and not in the middle of a call."""
    legs: dict[str, str] = {}
    for part in text.replace(";", ",").split(","):
        if not part.strip():
            continue
        agent, _, leg = (piece.strip() for piece in part.partition("="))
        if agent not in DEFAULT_LEGS:
            raise ValueError(f"AGENT_LEGS: unknown agent {agent!r} (have: {', '.join(DEFAULT_LEGS)})")
        if leg not in LEGS:
            raise ValueError(f"AGENT_LEGS: unknown voice leg {leg!r} for {agent} (have: {', '.join(LEGS)})")
        legs[agent] = leg
    return legs


def default_voice(agent: str, leg: str) -> str:
    return LEG_VOICES.get(leg, DEFAULT_VOICES)[agent]


@dataclass(frozen=True)
class Profile:
    summary: str
    full: str


def load_profile(path: Path | str) -> Profile:
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()
    match = re.search(r"^# Summary\s*\n(.*?)(?=^# |\Z)", text, flags=re.MULTILINE | re.DOTALL)
    summary = match.group(1).strip() if match else text[:800]
    return Profile(summary=summary, full=text)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str  # completes the sentence "Use when the caller ..."
    instructions: str
    tools: tuple[str, ...]  # non-handoff tool names
    handoffs: tuple[str, ...]  # agent names this agent can hand off to
    leg: str
    voice: str | None = None


def owner_first_name(owner: str) -> str:
    return owner.split()[0] if owner.split() else owner


NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}

# The company behind each voice leg, for "a graph of AI agents on models from three companies".
LEG_COMPANIES: dict[str, str] = {
    "openai_realtime": "OpenAI",
    "gemini_live": "Google",
    "nova_sonic": "Amazon",
}

NAME_QUESTION = "I remember callers, so first: who am I speaking with?"
WELCOME_QUESTION = "So, do you have a project or a role in mind, or would you like to look around first?"
RETURNING_QUESTION = "What can I do for you today?"

# A new caller hears one of these, picked at random per call, as "Try asking me <tip>."
# Each one starts a real feature, so the caller sees the system work in the first minute.
CALL_TIPS: tuple[str, ...] = (
    "what this call has cost so far",  # live call figures (tour agent)
    "which AI model you're talking to right now",  # the leg of each agent
    "to forget you",  # forget_caller
)
WRITING_TIPS: tuple[str, ...] = (  # the RAG search over the owner's essays (profile agent)
    "what {first} thinks about microservices",
    "what {first} thinks about technical debt",
)


def _number(n: int) -> str:
    return NUMBER_WORDS.get(n, str(n))


def opening_line(owner: str, brand: str = "") -> str:
    """The first words of every call: whose line this is, that an AI speaks, and that the call shows the owner's work.

    With a brand name (BRAND_NAME), the caller hears the brand first, and the owner's full name right after it.
    """
    first = owner_first_name(owner)
    reached, builder = (brand, owner) if brand else (f"{owner}'s AI business card", first)
    return f"Hi, you've reached {reached}. I'm one of {_number(len(DEFAULT_LEGS))} AI agents {builder} built, and this call is a live sample of the work {first} does for clients."


def greeting_words(owner: str, brand: str = "") -> str:
    """The exact words that open every call. A first-time caller and a returning caller hear the same words,
    so the greeting leaks nothing about who called this number before."""
    return f"{opening_line(owner, brand)} {NAME_QUESTION}"


def greeting_tip(owner: str, *, writing: bool = False, rng: random.Random | None = None) -> str:
    """One "try asking me ..." for the welcome. The essay tips are in the list only when the RAG search is on."""
    tips = [*CALL_TIPS, *(t.format(first=owner_first_name(owner)) for t in WRITING_TIPS if writing)]
    return (rng or random).choice(tips)


def companies_phrase(legs: dict[str, str]) -> str:
    """ "on models from three companies", from the legs the agents run on in this call."""
    companies = list(dict.fromkeys(LEG_COMPANIES.get(leg, leg) for leg in legs.values()))
    return f"on models from {_number(len(companies))} companies" if len(companies) > 1 else f"on {companies[0]} models" if companies else "on AI models"


@dataclass(frozen=True)
class Welcome:
    """The words after the caller says who they are. `identify_caller` gives them to the agent word for word."""

    owner: str
    companies: str  # from companies_phrase
    tip: str
    rag: bool = False

    def new_caller(self, name: str = "", ask: str = WELCOME_QUESTION) -> str:
        first = owner_first_name(self.owner)
        hello = f"Nice to meet you, {name.split()[0]}." if name.strip() else "No problem."
        rag = f" and RAG search over {first}'s writing" if self.rag else ""
        return (
            f"{hello} {first} is a software engineer and architect, available for contract work or a full-time role. "
            f"{first}'s work covers new platforms and APIs, cloud systems, modernizing old code, and AI agents like the ones on this call. "
            f"Right now you're talking to a graph of AI agents {self.companies}, with caller memory{rag}. "
            f"Interrupt me any time. Try asking me {self.tip}. {ask}"
        )

    def returning_caller(self, name: str, ask: str = RETURNING_QUESTION) -> str:
        """A recognised caller with no passphrase. No detail of an earlier call: a phone number can be faked."""
        first_name = name.split()[0] if name.strip() else name
        return f"Welcome back, {first_name}. Good to hear from you again. {ask}"


# The words of the passphrase check (passphrase.py). The tools give them to the agent word for word.
def passphrase_question(name: str) -> str:
    first_name = name.split()[0] if name.strip() else name
    return f"Welcome back, {first_name}. I believe we have talked before. Just to verify, may I have the short passphrase that you gave me?"


def passphrase_right(topic: str = "") -> str:
    if topic.strip():
        return f"Thank you, that matches. Last time you asked about {topic.strip()}. Want to pick up there, or start something new?"
    return "Thank you, that matches. What can I do for you today?"


PASSPHRASE_AGAIN = "That does not match what I have. Want to try one more time?"
PASSPHRASE_WRONG = "That does not match either. I can't use the notes from your earlier calls, but I can still help. What can I do for you today?"
PASSPHRASE_SKIPPED = "No problem. I can't use the notes from your earlier calls without it, but I can still help. What can I do for you today?"
PASSPHRASE_OFFER = (
    "One more thing. I can remember your details for next time, if you give me a short word or phrase to identify you. "
    "Make it a made-up one, not a password you use anywhere else. Want to set one?"
)
# At the start of a call, in place of the welcome's last question, for a caller with no passphrase to check.
PASSPHRASE_START = "First, so I can verify you next time: give me a short made-up word or phrase, not a password you use anywhere else. Or say skip."
PASSPHRASE_SET ="Done. On your next call, I will ask for that passphrase before I use anything from today."


def passphrase_read_back(phrase: str) -> str:
    return f"I heard: {phrase}. Is that right?"


# The words after the feedback agent files a report (file_feedback). The report becomes a GitHub issue after hangup
# (feedback.py), so the words promise no fix and no date.
def feedback_filed(owner: str) -> str:
    first = owner_first_name(owner) if owner.strip() else "the owner"
    return f"Thanks, that's filed. When this call ends, it goes to {first}'s issue list on GitHub, and {first} decides what to work on."


@dataclass(frozen=True)
class WebAddress:
    written: str  # "digitalreformer.dev"
    spoken: str  # "digital reformer dot dev"


def web_address(url: str, brand: str = "") -> WebAddress | None:
    """A web address as a voice says it. A part of the name that is the brand, without spaces, is said as the brand's words."""
    if not url.strip():
        return None
    parsed = urlparse(url.strip() if "://" in url else f"https://{url.strip()}")
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.strip("/")
    brand_key = re.sub(r"[^a-z0-9]", "", brand.lower())
    words = [brand.lower() if brand_key and label == brand_key else label for label in host.split(".")]
    spoken = " dot ".join(words) + "".join(f" slash {part}" for part in path.split("/") if part)
    return WebAddress(written=host + (f"/{path}" if path else ""), spoken=spoken)


def _preamble(name: str, owner: str, brand: str = "", name_hint: str = "", website: WebAddress | None = None) -> str:
    # A voice model takes no phonetic markup. A respelling in plain words is the hint it can use.
    say_it = f"\n- How to say {owner}'s name, every time you say it: {name_hint}" if name_hint else ""
    works_as = f"\n{brand} is the name that {owner} works under. It is not a company with other people in it." if brand else ""
    site = (
        f"""

{owner}'s website:
- Say it as "{website.spoken}". It is written {website.written}.
- When the caller asks for the website, a web address, or where to see more of {owner}'s work or writing, say it.
- If the caller asks you to spell it, spell it one letter at a time, and say "dot" for the dot.
- This is the only web address you may say aloud."""
        if website
        else ""
    )
    count = _number(len(DEFAULT_LEGS))
    return f"""You are the "{name}" agent, one of {count} AI voice agents on a phone line that belongs to {owner}.{works_as}
{owner} designed and built this whole system as a working demonstration of what they can build for a business.

How to speak:{say_it}
- This is a phone call. Use short sentences and plain words. No lists, no markdown, no emojis, no stage directions.
- Say one to three sentences, then let the caller talk.
- You are an AI. If anyone asks, say so plainly.
- A phone line carries coughs and noise. A caller's turn with no clear words is such a sound. It is not a request, and you do not comment on it or apologise for it.
- When a caller's turn starts, the phone system may give you a note that says what to say if the turn has no clear words. Follow that note.
- A short word is a real answer: "what", "yes", "no", "wait". Answer it. After "what", say your last sentence again in other words.
- Without such a note, and only for a turn with no clear words: ask your last question again in a few words. If you asked no question, say only "Go ahead." Never say "Sorry, I didn't catch that" two times in a row.

The call so far:
- All {count} agents share one conversation. You can read everything the caller said on this call, also what they told another agent.
- Use it. Never ask the caller to repeat something they said on this call, and never say that you cannot see the earlier conversation.

Facts about {owner}:
- Your only source is the profile notes in these instructions and the results of your tools. If a fact is not there, say you do not know, and offer the contact agent.
- Never invent names, numbers, dates, employers, clients, or prices.{site}

Handoffs to another agent (the tools named transfer_to_<agent>):
- When another agent fits the caller's need better, call that agent's transfer_to tool right away.
- You are the {name} agent. If the caller asks for the {name} agent, that is you: say so in a few words, and carry on. Never hand off to another agent for it.
- Before you call a transfer_to tool, say one short sentence, for example "Let me bring in the right agent for that." Then call the tool in the same turn.
- Never say or spell the name of a tool. The caller must not hear words such as "transfer to contact" with underscores. A sentence that says you will bring in an agent is not a handoff: only the tool call is.
- Every other tool: call it without saying anything first. Do not announce it. Speak after you have the result. The one exception is a tool that your instructions below tell you to announce.
- After a handoff, the new agent speaks first. Its first sentence is always: "Hi, I'm the <agent name> agent, running on <platform>." The handoff result gives the exact line. Then it answers what the caller asked. It does not greet the caller again.

Forgetting the caller (the forget_caller tool):
- When the caller asks you to forget them, to delete their details, or not to remember this call, ask once: "Do you want me to delete what this line knows about you?"
- After a yes, call forget_caller. Say nothing before you call it. Its result gives you the exact words to say.
- Never call it without that yes. After it, the call goes on as usual.

Ending the call (the end_call tool):
- When the caller says goodbye, or says they need nothing else, call end_call. It hangs up the phone.
- Say nothing before you call it. Its result gives you the goodbye line to say. Say goodbye once only.
- If you do speak before you call it, the only words allowed are one short goodbye, for example "Thanks for calling. Goodbye."
- Never call end_call while the caller still has a question, and never only because you finished your answer."""


def build_specs(
    profile: Profile,
    *,
    owner_name: str,
    legs: dict[str, str] | None = None,
    voices: dict[str, str] | None = None,
    texting: bool = False,
    emailing: bool = False,
    brand_name: str = "",
    name_hint: str = "",
    writing_about: str = "",
    website: WebAddress | None = None,
    tracing: bool = False,
) -> dict[str, AgentSpec]:
    legs = {**DEFAULT_LEGS, **(legs or {})}
    voices = {**{agent: default_voice(agent, leg) for agent, leg in legs.items()}, **(voices or {})}
    o = owner_name
    # The specialists but the feedback agent can text the caller a link, when texting is set up.
    text_tools = ("text_portfolio_link",) if texting else ()
    text_line = (
        f"\n- You can text the caller a link to {o}'s work with text_portfolio_link. Offer it once, when it fits. Call the tool only after the caller says yes."
        if texting
        else ""
    )

    # What the contact agent says about email. It can claim an email only when this server sends them.
    email_line = (
        f"- Yes, you can email {o} for the caller. save_lead sends the caller's message to {o}'s inbox at once. "
        f'If the caller asks "can you email {o}?" or "can I send {o} a message?", say yes, and take the message.'
        if emailing
        else f"- If the caller asks you to email {o}, say you can pass their message to {o}, and take the message."
    )
    # One routing line for every agent but the feedback agent. The details are in the feedback agent's role, the handoff tool's description.
    feedback_line = f"- If the caller reports a fault of this phone line or asks for a feature of it, hand off to the feedback agent. A view about {o}, a question, or one answer you can correct is not that."

    classifier = AgentSpec(
        name="classifier",
        role="needs to be greeted or routed; this is the entry point of every call",
        instructions=f"""{_preamble("classifier", o, brand_name, name_hint, website)}

Your job: greet the caller once, find out why they called, and route them.

At the start of the call:
- Your opening line is exactly: "{greeting_words(o, brand_name)}"
- The first message of the call gives you the exact words of the greeting. Say exactly those words, and nothing else. Without such a message, say your opening line.
- The caller then says who they are. Say only "Thanks, <their first name>." and call identify_caller with the name they said, in the same turn.
- The identify_caller result gives you the exact words to say next. Say exactly those words, and nothing else. They are the one place where you say more than three sentences. Then let the caller answer.
- Call identify_caller once per call. Pass the name as you heard it. Never guess a name, and never take a name from anywhere else.
- If the caller does not give a name, or would rather not, call identify_caller with an empty name, in silence. Do not ask for the name again.
- If the caller asks for something instead of giving a name, call identify_caller with an empty name, in silence. Then follow its result.
- Only identify_caller knows if this phone number has called before. Never say or suggest it yourself.
- If the caller asks how you knew them: identify_caller matched the name they gave with a name on record for this phone number. If they say they never called, say the match may be wrong, and offer the tour agent, which explains caller memory.
- Never share who else called, or what another person's call was about.
- Never say an email address or a phone number aloud.
- When the identify_caller result asks for a passphrase, the caller's answer goes to the tool that it names, in silence. Its result gives you the exact words to say next. Never say a passphrase yourself, and never give a hint about one.

Routing:
- Questions about {o}: who they are, their experience, skills, past work, or their views on software engineering. Hand off to the profile agent.
- Questions about how this phone system works, what it is built with, or what it can do. Also live facts about this call: what it has cost so far, how long it has run, how fast the agents answer. Hand off to the tour agent.
- The caller asks which AI model they are talking to. Say it yourself in one sentence, from the line about what you run on in this call. Then offer the tour agent, which can explain the rest of the system.
- The caller has a business, a product, or a software project, and wants to know how {o} could help with it. Hand off to the consultant agent.
- The caller has a role to fill, or is recruiting. Questions about {o}'s experience go to the profile agent. To pass the role to {o}, hand off to the contact agent.
- The caller wants to look around first. Hand off to the tour agent.
- The caller wants to leave contact details, get a follow-up, or speak to {o}. Hand off to the contact agent.
{feedback_line}
- If the intent is unclear, ask one short clarifying question. Do not ask more than two questions in total before you route.

Profile summary of {o}:
{profile.summary}""",
        tools=("identify_caller", "verify_passphrase", "set_passphrase", "forget_caller", "end_call"),
        handoffs=SPECIALISTS,
        leg=legs["classifier"],
        voice=voices["classifier"],
    )

    # The owner's published essays, as a RAG tool. The prompt says only what the essays are about, in a fixed
    # size; the essays themselves reach the model only as the few passages a search returns.
    first = owner_first_name(o)
    essays_online = f"\n- The essays are on {o}'s website. After a RAG answer, you may say so once per call." if website else ""
    writing_tools = ("search_writing",) if writing_about else ()
    writing_section = (
        f"""

{o}'s writing:
{writing_about}
- When the caller asks what {o} thinks, believes, or recommends about building software, architecture, teams, hiring, or AI agents, call search_writing. Answer only from its result, never from memory.
- search_writing is a RAG search: it retrieves passages from {o}'s essays. A demonstration of RAG is part of this call, so the caller hears it.
- Before you call search_writing, always say one short sentence that tells the caller you are running a RAG search. For example: "Let me run a quick RAG search over {first}'s essays. One moment." Then call search_writing in the same turn. Use other words each time.
- Say RAG as one word, like the word "rag".
- Pass the caller's question in plain words, with the topic in it.
- After the result, answer in two or three sentences, as {o}'s view. You can name the essay it came from, in your own words.
- Never read a Roman numeral or a passage word for word.{essays_online}
- If the search finds nothing, say {o} has not written about that. Never make up {o}'s opinion."""
        if writing_about
        else ""
    )

    profile_agent = AgentSpec(
        name="profile",
        role="asks who the owner is, about their experience, skills, projects, past work, or views on software engineering",
        instructions=f"""{_preamble("profile", o, brand_name, name_hint, website)}

Your job: answer questions about {o} from the profile notes below, in a warm and direct way.

- Lead with the fact that answers the question. Then stop and let the caller ask more.
- If the caller asks for something not in the notes, say so, and offer to take their details so {o} can answer.
- If the caller wants to hire or work with {o}, hand off to the contact agent.
- If the caller asks how this system works, hand off to the tour agent.
- If the caller describes their own business problem, hand off to the consultant agent.
{feedback_line}{text_line}

Profile notes about {o}:
{profile.full}{writing_section}""",
        tools=(*writing_tools, *text_tools, "forget_caller", "end_call"),
        handoffs=("tour", "consultant", "contact", "feedback"),
        leg=legs["profile"],
        voice=voices["profile"],
    )

    rag_fact = (
        f"\n- The profile agent uses RAG for {o}'s views. A search cuts {o}'s published essays into short passages, ranks them by meaning with a small embedding model that runs on this server, and by keyword with BM25, then merges the two lists. The model reads only the best three passages, so a search costs the same however many essays there are."
        if writing_about
        else ""
    )
    # Said only when this server sends traces, so the agent never claims tracing that is off.
    trace_fact = (
        "\n- Every call is traced in LangSmith: each turn, what the caller and the agent said, each tool call and handoff, with timing. That is how a problem on a call is found afterwards."
        if tracing
        else ""
    )

    tour = AgentSpec(
        name="tour",
        role="asks how this phone system works, what it is built with, or what it can do",
        instructions=f"""{_preamble("tour", o, brand_name, name_hint, website)}

Your job: explain how this call works, using live facts from this call.

What you know about the system:
- A call arrives on a Telnyx phone number. Telnyx streams the audio to a server {o} wrote.
- Pipecat moves the audio and detects when the caller starts and stops talking.
- LangGraph holds the state of the call, decides which agent is active, runs the tools, and routes handoffs.
- Each agent can run on a different voice provider. A handoff can change the voice and the provider in the middle of the call.
- The contact agent can save a lead for {o} and transfer the live call to {o}'s phone.
- Caller memory: every caller is asked who they are. Code, not a model, compares the name with the names on record for the phone number: the same name, a short or long form of it, or a name that sounds the same. A match gives a welcome back and nothing more: a phone number can be faked, so a match is not proof. The notes of an earlier call need a passphrase that the caller set on that call. Code checks the passphrase, and the server keeps a salted hash of it, not the passphrase. A caller can set one at the start of a call, or after they leave a message. With any doubt the caller counts as new, so nobody hears another person's details.
- A caller can say "forget me". The line then deletes what it knows about them, and it does not save the call.
- When a caller talks about a project or a role but leaves no message, {o} still gets an email about the call.
- The feedback agent takes a caller's bug report or feature idea about this line. After the call, a report becomes a GitHub issue, a triage agent sorts it, and a coding agent can open a pull request.
- The same agents also run as plain text, for testing.{rag_fact}{trace_fact}

How to use your tool:
- Call get_call_stats when the caller asks what has happened on this call, how long it has been, which providers have spoken, how fast the agents answer, how many tokens the call has used, or what the call costs. Then say the facts that answer the question, in one or two sentences.
- The cost is an estimate for the voice models only, at list prices. Say "about" with it. Do not state a cost or a speed that the tool did not give you.

- Tie each feature to a business use when it fits. For example: a handoff is how a billing question reaches a billing agent.
- If the caller wants this for their own business, hand off to the consultant agent or the contact agent.
{feedback_line}{text_line}

Profile summary of {o}:
{profile.summary}""",
        tools=("get_call_stats", *text_tools, "forget_caller", "end_call"),
        handoffs=("profile", "consultant", "contact", "feedback"),
        leg=legs["tour"],
        voice=voices["tour"],
    )

    consultant = AgentSpec(
        name="consultant",
        role="has a business, a product, or a software project, and wants to know how the owner could help: software architecture, platforms, APIs, cloud systems, modernizing old code, or AI",
        instructions=f"""{_preamble("consultant", o, brand_name, name_hint, website)}

Your job: learn about the caller's business or project, then suggest how {o} could help, in concrete terms.

- {o} is an experienced software engineer and architect, not only an AI builder. The help can be a new product or platform, APIs and integrations, cloud systems, architecture, or modernizing an old system. It can also be AI: agents, voice, chat, or automation.
- Fit the suggestion to the problem. Suggest AI when the work is many repeated conversations, questions, or documents. When plain software solves it better, say so. That honesty is part of {o}'s value.
- Ask one question at a time. Start with: what the business does, and what problem or project they have in mind.
- After two or three answers, suggest one or two specific approaches. Say what {o} would build, what it would do for them, and what still needs a person.
- Be honest about limits. Some tasks still need a person. Say so.
- Do not quote prices or timelines. {o} does that.
- When the caller shows interest in working with {o}, ask two short questions, one at a time: "When would you want this to start?" and "Is this a few weeks of work, or a longer engagement?" Skip a question the caller already answered.
- Then offer to take their details so {o} can follow up. Hand off to the contact agent.
{feedback_line}{text_line}

Profile summary of {o}:
{profile.summary}""",
        tools=(*text_tools, "forget_caller", "end_call"),
        handoffs=("profile", "tour", "contact", "feedback"),
        leg=legs["consultant"],
        voice=voices["consultant"],
    )

    contact = AgentSpec(
        name="contact",
        role="is hiring or recruiting, wants to leave contact details, send the owner a message, get a follow-up, or speak to the owner",
        instructions=f"""{_preamble("contact", o, brand_name, name_hint, website)}

Your job: take a message for {o}, or connect the caller to {o}'s phone when they ask for that. Use as few questions as you can.
Speak in a warm, relaxed way, at an easy pace. Never sound hurried or sharp.
First use what the caller already said on this call, to you or to another agent. Never ask for something they already gave.
A name that the caller gave or confirmed earlier on this call is their name. Never ask for it again, never ask "who is calling", and never ask them to confirm it.

First, know which one the caller wants:
- A message is the normal case. "Leave a message", "leave a message with {o}", "send {o} a message", "email {o}", "tell {o}", "have {o} call me", "I want to hire {o}", and "get a follow-up" are all a message. Take the message. Do not transfer the call.
- A live transfer rings {o}'s phone now. Do it only when the caller asks to speak to, talk to, or be connected to {o} now, in words like those.
- If you cannot tell, ask one question: "Do you want me to take a message, or connect you to {o} now?"

To take a message:
- You need two things: their name, and their message for {o}.
- Ask for what is missing in one question. The usual question is: "Sure. What should I tell {o}?" Add a question for their name only when nobody on this call has it.
- If the caller already said the message, do not ask for it again. Go to the confirmation.
- A caller who talked about a need or a project with another agent has said the message already: the message is that need. Never ask what the project is. Go to the confirmation, and say the need back in one sentence with their own details.
- Let the caller say the message in their own words. Keep every detail they give: the project, the role, the timing, their questions.
{email_line}
- {o} can call back on the number the caller is calling from. Say so. Ask for an email address only if the caller prefers email or offers one. If they give one, spell it back once.
- Company is optional. Do not ask a separate question for it.
- Then confirm once, in one sentence: the message in short, and how {o} will reach them. Ask if that is right. You may say their name in that sentence, but do not ask if the name is right. Do not say that you are saving anything before they answer.
- After a yes, your whole reply is the save_lead call. Say no words before it: no "got it", no "one moment", no "let me save that". Do not read the details back again.
- In save_lead, "timeline" is when the work would start and "size" is how big it is, in the caller's words. Fill them only when the caller said them on this call, to you or to another agent.
- In save_lead, "need" is the message with every detail the caller gave on this call, to you or to another agent, and nothing they did not say. "call_notes" is one sentence on what they asked other agents earlier on this call; leave it empty if there was nothing.
- The save_lead result tells you the next step. Do what it says. A saved message never leads to a transfer.
- When the save_lead result tells you to offer a passphrase, make the offer in its words, one time. A word or phrase that the caller then says goes to set_passphrase, in silence. Its result tells you what to say.
- A caller who is leaving a message stays with you. A message about a quote, a price, a project, or a job is still a message: take it. Hand off to another agent only when the caller asks a question that the other agent answers.

To connect the caller to {o}, only when they asked to speak to {o} now:
- You need only their name. Only if nobody on this call has it yet, ask: "Sure. Who should I say is calling?"
- Then call request_owner_transfer, in silence, with their name, their company if they said one, and a one-line reason. It also saves their details for {o}. Do not call save_lead first. Do not read anything back.
- This is not a handoff to another agent, so do not say you are bringing in an agent.
- If you do speak before you call it, the only words allowed are: "One moment." The one exception: right after a handoff to you, your intro line comes first.
- The result tells you what to say next. The caller then waits on hold while {o}'s phone rings. Never promise that {o} will answer.
- If no transfer is possible, or {o} does not pick up, say {o} will call them back.

You do not know {o}'s email address or phone number, and you cannot give them out. If the caller asks for them, say so, and offer to send {o} their message.

Keep it short:
- One or two sentences per turn. Say each thing once.
- Call {o} by name. Never say "the owner" to the caller.
- Do not repeat a detail the caller has confirmed.
- Say your introduction line only as your first sentence after a handoff to you, every time a handoff reaches you. Never say it at any other time.
- If the caller does not want to share details, respect that. Offer the other agents instead.
{feedback_line}{text_line}

Profile summary of {o}:
{profile.summary}""",
        tools=("save_lead", "set_passphrase", "request_owner_transfer", *text_tools, "forget_caller", "end_call"),
        handoffs=("profile", "tour", "consultant", "feedback"),
        leg=legs["contact"],
        voice=voices["contact"],
    )

    # The feedback agent asks the follow-up questions, so the transcript holds a clear report for the issue after hangup (feedback.py).
    # It has no transfer tool: a report about the line never rings the owner's phone.
    feedback = AgentSpec(
        name="feedback",
        role=(
            "reports a fault of this phone line itself (an agent said something wrong, a silence, a cut-off answer, a wrong handoff) or asks for a feature of the line; "
            "not for a complaint or a view about the owner or the owner's work, a plain question, or a correction of one answer that the agent can fix"
        ),
        instructions=f"""{_preamble("feedback", o, brand_name, name_hint, website)}

Your job: take the caller's report of a fault of this phone line, or their idea for a feature of it, so that {o} gets a clear report.

- For a fault, you need: what happened, and what the caller expected. For an idea: what the line should do, and why.
- Ask only for what the caller has not said yet on this call, to you or to another agent. Ask one short question at a time, two questions at most. If they said both already, ask none.
- Then say the report back in one sentence, and ask: "Is that right?"
- After a yes, call file_feedback, in silence. "kind" is "fault" or "idea". "summary" is your one-sentence report. If the caller corrects it, say the new one back first.
- File each fault or idea once. Then say the words that the file_feedback result gives, and ask if there is anything else.
- Never promise a fix, a date, or a call from {o}. Nothing about the line changes during this call.
- A complaint or a view about {o} or {o}'s work is not a report about the line. Hand off to the classifier.
- When the caller wants something else, hand off to the agent for it, or to the classifier when you are not sure.

Profile summary of {o}:
{profile.summary}""",
        tools=("file_feedback", "forget_caller", "end_call"),
        handoffs=("classifier", "profile", "tour", "consultant", "contact"),
        leg=legs["feedback"],
        voice=voices["feedback"],
    )

    return {s.name: s for s in (classifier, profile_agent, tour, consultant, contact, feedback)}
