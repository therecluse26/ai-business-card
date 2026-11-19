"""The owner's published writing as a RAG tool: feed, passages, hybrid search, saved vectors, and the prompts."""

from __future__ import annotations

import re

import numpy as np
from langchain_core.messages import HumanMessage, ToolMessage

from ai_call_center.agents.specs import Profile, build_specs
from ai_call_center.graph import tools as graph_tools
from ai_call_center.writing import RESULT_PASSAGES, Library, about_line, load_writing, parse_feed, passages_of, search_result
from tests.conftest import OWNER, scripted, tool_call

FEED = """<?xml version="1.0"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel><title>Theses</title>
<description>What I believe about building software, and why.</description>
<item><title>VI. Code that is hard to test is coupled code.</title><link>https://example.dev/vi/</link>
<description>A test that needs a database to check a price rule is telling you something.</description>
<content:encoded><![CDATA[<p>When a unit test needs half the system, the design is coupled.</p>
<pre class="astro-code"><code><span>const db = connect()</span></code></pre>
<h2 id="what">What to do</h2><p>Pass dependencies in &amp; keep rules pure.</p>]]></content:encoded></item>
<item><title>VII. Microservices too early and a monolith split too late are the same mistake.</title><link>https://example.dev/vii/</link>
<description>Service boundaries follow the team and the domain.</description>
<content:encoded><![CDATA[<p>Split a monolith when teams step on each other, not before.</p>]]></content:encoded></item>
<item><title>XXIV. Hire for judgment. Syntax is now free.</title><link>https://example.dev/xxiv/</link>
<description>Interview for decisions, not trivia.</description>
<content:encoded><![CDATA[<p>Agents write the syntax. People still decide what to build.</p>]]></content:encoded></item>
</channel></rss>"""

# A stand-in for the embedding model: words with the same meaning share one dimension.
MEANINGS = [
    {"test", "tests", "testing", "testable", "coupled", "dependencies"},
    {"microservices", "monolith", "service", "split", "architecture"},
    {"hire", "hiring", "interview", "recruit", "judgment", "candidates"},
]


class FakeEmbedder:
    name = "fake"

    def __init__(self) -> None:
        self.embedded = 0

    def _vector(self, text: str) -> np.ndarray:
        words = set(re.findall(r"[a-z]+", text.lower()))
        v = np.array([len(words & group) for group in MEANINGS] + [0.1], dtype=np.float32)
        return v / np.linalg.norm(v)

    def passages(self, texts: list[str]) -> np.ndarray:
        self.embedded += len(texts)
        return np.stack([self._vector(t) for t in texts])

    def query(self, text: str) -> np.ndarray:
        return self._vector(text)


def library(embedder=None, vector_cache=None) -> Library:
    return Library(parse_feed(FEED), embedder=embedder, vector_cache=vector_cache)


def test_a_post_is_plain_text_without_code_blocks():
    writing = parse_feed(FEED)
    post = writing.posts[0]
    assert writing.about == "What I believe about building software, and why."
    assert post.title == "VI. Code that is hard to test is coupled code."
    assert "connect()" not in post.text  # code cannot be spoken
    assert "<" not in post.text
    assert "Pass dependencies in & keep rules pure." in post.text


def test_a_heading_starts_a_new_passage_and_a_long_section_is_cut():
    post = parse_feed(FEED).posts[0]
    passages = passages_of(post)
    assert [p.text.splitlines()[0] for p in passages] == [post.summary, "What to do:"]
    long = passages_of(post, max_chars=60)
    assert len(long) > len(passages)
    assert all(p.title == post.title for p in long)


def test_meaning_finds_a_post_that_shares_no_keyword():
    question = "Who should I recruit?"
    assert library().search(question) == []  # keywords alone find nothing
    hits = library(FakeEmbedder()).search(question)
    assert hits[0].passage.title.startswith("XXIV.")
    assert hits[0].keyword_rank is None and hits[0].meaning_rank == 1


def test_a_passage_found_by_both_rankings_comes_first():
    hits = library(FakeEmbedder()).search("When do we split the monolith?")
    assert hits[0].passage.title.startswith("VII.")
    assert hits[0].keyword_rank == 1 and hits[0].meaning_rank == 1


def test_a_question_the_writing_does_not_cover_gets_no_passage():
    lib = library(FakeEmbedder())
    assert lib.search("favourite pizza topping") == []
    assert "found no passage about this question" in search_result(lib, "favourite pizza topping")


def test_the_model_reads_a_few_passages_never_the_whole_feed():
    lib = library(FakeEmbedder())
    result = search_result(lib, "testing coupled code, microservices, and hiring")
    assert result.startswith("RAG search (hybrid: semantic embeddings and keyword BM25) retrieved")
    assert result.count('From the essay titled "') <= RESULT_PASSAGES
    assert "Do not read them aloud." in result


def test_saved_vectors_are_reused_and_only_new_passages_are_embedded(tmp_path):
    cache = tmp_path / "vectors.npz"
    first = FakeEmbedder()
    library(first, cache)
    assert first.embedded == len(library().passages)
    again = FakeEmbedder()
    library(again, cache)
    assert again.embedded == 0


def test_the_saved_copy_is_used_when_the_feed_cannot_be_read(tmp_path, monkeypatch):
    cache = tmp_path / "writing.json"
    monkeypatch.setattr("ai_call_center.writing.fetch_feed", lambda url, timeout=5.0: FEED.encode())
    assert len(load_writing("https://example.dev/rss.xml", cache).posts) == 3

    def offline(url, timeout=5.0):
        raise OSError("network is down")

    monkeypatch.setattr("ai_call_center.writing.fetch_feed", offline)
    assert load_writing("https://example.dev/rss.xml", cache) == parse_feed(FEED)
    assert load_writing("https://example.dev/rss.xml", tmp_path / "missing.json").posts == []


def _specs(writing_about: str = ""):
    profile = Profile(summary=f"{OWNER} builds AI systems.", full=f"# Summary\n{OWNER} builds AI systems.")
    return build_specs(profile, owner_name=OWNER, writing_about=writing_about)


def test_the_profile_agent_announces_a_rag_search_and_sees_no_essay_text():
    specs = _specs(about_line(library(), OWNER))
    profile = specs["profile"]
    assert "search_writing" in profile.tools
    assert f"{OWNER} has published 3 short essays on software engineering." in profile.instructions
    assert "Let me run a quick RAG search" in profile.instructions
    assert "Hire for judgment" not in profile.instructions  # essays reach the model only through a search
    assert "uses RAG" in specs["tour"].instructions
    assert all("search_writing" not in s.tools for name, s in specs.items() if name != "profile")


def test_without_a_feed_there_is_no_rag_tool():
    specs = _specs()
    assert "search_writing" not in specs["profile"].tools
    assert "RAG" not in specs["profile"].instructions
    assert "RAG" not in specs["tour"].instructions


async def test_search_writing_runs_in_the_graph(make_graph, ctx, config, monkeypatch):
    monkeypatch.setattr(graph_tools, "default_library", lambda: library(FakeEmbedder()))
    model = scripted(tool_call("search_writing", query="When do you split a monolith?"), f"{OWNER} waits until teams get in each other's way.")
    graph = make_graph(model)
    result = await graph.ainvoke(
        {"messages": [HumanMessage("When do you split a monolith?")], "active_agent": "profile", "handoffs": ["profile"]},
        config=config,
        context=ctx,
    )
    found = next(m.content for m in result["messages"] if isinstance(m, ToolMessage))
    assert 'From the essay titled "VII.' in found


def test_a_web_address_is_said_in_words():
    from ai_call_center.agents.specs import web_address

    site = web_address("https://www.digitalreformer.dev/", "Digital Reformer")
    assert site.written == "digitalreformer.dev"
    assert site.spoken == "digital reformer dot dev"
    assert web_address("example.com/work/", "").spoken == "example dot com slash work"
    assert web_address("") is None


def test_every_agent_can_give_the_website_and_the_goodbye_names_it():
    from ai_call_center.agents.specs import web_address
    from ai_call_center.graph.tools import SAY_GOODBYE, say_goodbye

    profile = Profile(summary=f"{OWNER} builds AI systems.", full=f"# Summary\n{OWNER} builds AI systems.")
    site = web_address("https://digitalreformer.dev", "Digital Reformer")
    specs = build_specs(profile, owner_name=OWNER, website=site, writing_about=about_line(library(), OWNER))
    for spec in specs.values():
        assert 'Say it as "digital reformer dot dev". It is written digitalreformer.dev.' in spec.instructions
        assert "web address aloud" not in spec.instructions.replace("only web address you may say aloud", "")
    assert "The essays are on" in specs["profile"].instructions
    assert "digital reformer dot dev" in say_goodbye(site.spoken)
    assert say_goodbye("") == SAY_GOODBYE
    no_site = build_specs(profile, owner_name=OWNER)
    assert "website" not in no_site["contact"].instructions
