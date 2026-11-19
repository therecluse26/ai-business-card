"""The owner's published writing, from an RSS feed, as a RAG tool for the profile agent.

The server reads the feed once at start and keeps a copy on disk. When the feed cannot be read, the copy
is used, so a call never waits on the network.

Retrieval is hybrid. Each post is cut into short passages. A local embedding model (fastembed, ONNX on the
CPU) ranks the passages by meaning, BM25 ranks them by keyword, and reciprocal rank fusion merges the two
lists. The model reads only the best few passages, so the cost of a search stays the same as posts are added.
A passage's vector is saved on disk under a hash of its text: a restart embeds only new or changed passages.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}encoded"
FETCH_TIMEOUT_SECS = 5.0
PASSAGE_CHARS = 900  # a passage is about one section of a post
RESULT_PASSAGES = 3  # the model reads this many passages per search, at most
PASSAGES_PER_POST = 2  # so one long post cannot fill the whole result
# Cosine similarity, measured on the real feed on 2026-11-17: topic questions reach 0.62 to 0.79,
# off-topic ones ("Do you like dogs?", "Is Brad good at golf?") reach 0.43 to 0.52.
# With the model loaded, a passage under this floor is out, also when it shares a keyword.
MIN_SIMILARITY = 0.55
RRF_K = 60  # the usual constant for reciprocal rank fusion

# Words that say nothing about the topic. The keyword search ignores them.
STOP_WORDS = frozenset(
    "a an and are as at be but by can do does for from has have he her his how i if in into is it its me my "
    "not of on or our she so that the their them then there they this to was we were what when where which "
    "who why will with would you your about think thinks believe believes view opinion say says".split()
)


@dataclass(frozen=True)
class Post:
    title: str
    link: str
    summary: str
    text: str


@dataclass(frozen=True)
class Writing:
    """The feed: what it says it is about, and its posts."""

    about: str = ""
    posts: list[Post] = field(default_factory=list)


@dataclass(frozen=True)
class Passage:
    title: str  # the post's title
    text: str

    @property
    def key(self) -> str:
        return hashlib.sha1(f"{self.title}\n{self.text}".encode()).hexdigest()

    def for_embedding(self) -> str:
        return f"{self.title}\n{self.text}"


def plain_text(fragment: str) -> str:
    """Spoken-ready text from a post's HTML. A code block cannot be spoken, so it is dropped.
    A section heading becomes a line that starts with "## ", so the passages can follow the sections."""
    fragment = re.sub(r"<pre\b.*?</pre>", " ", fragment, flags=re.DOTALL | re.IGNORECASE)
    fragment = re.sub(r"<h[1-6][^>]*>", "\n## ", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"</(p|h[1-6]|li|blockquote)>", "\n", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in html.unescape(fragment).splitlines())
    return "\n".join(line for line in lines if line and line != "##")


def parse_feed(xml: bytes | str) -> Writing:
    root = ET.fromstring(xml)
    posts = []
    for item in root.iter("item"):
        body = item.findtext(CONTENT_NS) or item.findtext("description") or ""
        posts.append(
            Post(
                title=(item.findtext("title") or "").strip(),
                link=(item.findtext("link") or "").strip(),
                summary=plain_text(item.findtext("description") or ""),
                text=plain_text(body),
            )
        )
    about = plain_text(root.findtext("channel/description") or "")
    return Writing(about=about, posts=[p for p in posts if p.title])


def fetch_feed(url: str, timeout: float = FETCH_TIMEOUT_SECS) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "voice-business-card/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_writing(url: str, cache_path: Path) -> Writing:
    """The feed, read now and saved to `cache_path`. When the feed fails, the saved copy. Else nothing."""
    if url:
        try:
            writing = parse_feed(fetch_feed(url))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            saved = {"about": writing.about, "posts": [asdict(p) for p in writing.posts]}
            cache_path.write_text(json.dumps(saved, indent=1), encoding="utf-8")
            log.info("writing: %d posts from %s", len(writing.posts), url)
            return writing
        except Exception as exc:  # noqa: BLE001 - any failure falls back to the saved copy
            log.warning("writing: cannot read the feed %s (%s); using the saved copy", url, exc)
    if cache_path.exists():
        try:
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            return Writing(about=saved["about"], posts=[Post(**row) for row in saved["posts"]])
        except Exception as exc:  # noqa: BLE001
            log.warning("writing: the saved copy %s is not usable (%s)", cache_path, exc)
    return Writing()


def passages_of(post: Post, max_chars: int = PASSAGE_CHARS) -> list[Passage]:
    """A post as passages. A heading starts a new passage; a long section is cut between paragraphs."""
    out: list[Passage] = []
    current: list[str] = [post.summary] if post.summary else []

    def flush() -> None:
        if current:
            out.append(Passage(post.title, "\n".join(current)))
            current.clear()

    for line in post.text.splitlines():
        is_heading = line.startswith("## ")
        size = sum(len(x) + 1 for x in current)
        if current and (is_heading or size + len(line) > max_chars):
            flush()
        current.append(line[3:] + ":" if is_heading else line)
    flush()
    return out


def _stem(word: str) -> str:
    if word.endswith("ss"):
        return word
    for suffix in ("ing", "ies", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    # "hire" and "hiring" meet at "hir"; "couple" and "coupled" at "coupl".
    return word[:-1] if word.endswith("e") and len(word) >= 4 else word


def terms(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.casefold().replace("’", "'")) if w not in STOP_WORDS and len(w) > 1]


class Embedder(Protocol):
    name: str

    def passages(self, texts: list[str]) -> np.ndarray: ...  # one unit-length row per text

    def query(self, text: str) -> np.ndarray: ...  # one unit-length vector


class FastEmbedder:
    """A small local embedding model (BAAI/bge-small-en-v1.5 by default: 384 numbers, about 65 MB)."""

    def __init__(self, model: str, cache_dir: Path) -> None:
        from fastembed import TextEmbedding

        self.name = model
        self.model = TextEmbedding(model, cache_dir=str(cache_dir))

    def passages(self, texts: list[str]) -> np.ndarray:
        return _unit(np.array(list(self.model.passage_embed(texts)), dtype=np.float32))

    def query(self, text: str) -> np.ndarray:
        return _unit(np.array(next(iter(self.model.query_embed([text]))), dtype=np.float32))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def embed_with_cache(embedder: Embedder, passages: list[Passage], cache_path: Path | None) -> np.ndarray:
    """One vector per passage. Saved vectors are reused; only new or changed passages are embedded."""
    saved: dict[str, np.ndarray] = {}
    if cache_path is not None and cache_path.exists():
        try:
            with np.load(cache_path) as data:
                if str(data["model"]) == embedder.name:
                    saved = dict(zip((str(k) for k in data["keys"]), data["vectors"]))
        except Exception as exc:  # noqa: BLE001
            log.warning("writing: the saved vectors %s are not usable (%s)", cache_path, exc)
    missing = [p for p in passages if p.key not in saved]
    if missing:
        saved.update(zip((p.key for p in missing), embedder.passages([p.for_embedding() for p in missing])))
        log.info("writing: embedded %d new passages of %d", len(missing), len(passages))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            keys = [p.key for p in passages]
            np.savez(cache_path, model=embedder.name, keys=np.array(keys), vectors=np.stack([saved[k] for k in keys]))
    return np.stack([saved[p.key] for p in passages]) if passages else np.zeros((0, 0), dtype=np.float32)


@dataclass(frozen=True)
class Hit:
    passage: Passage
    keyword_rank: int | None  # 1 is the best; None = no shared keyword
    meaning_rank: int | None  # None = no embedding model
    similarity: float | None


class Library:
    """Hybrid retrieval over the passages of every post."""

    def __init__(
        self,
        writing: Writing,
        *,
        embedder: Embedder | None = None,
        vector_cache: Path | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.writing = writing
        self.passages = [p for post in writing.posts for p in passages_of(post)]
        self.embedder = embedder
        self.vectors = embed_with_cache(embedder, self.passages, vector_cache) if embedder and self.passages else None
        # BM25. The post title is part of every passage: a title is the claim itself.
        self.k1, self.b = k1, b
        self.docs = [Counter(terms(p.title) + terms(p.text)) for p in self.passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.passages else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.passages)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @property
    def posts(self) -> list[Post]:
        return self.writing.posts

    @property
    def method(self) -> str:
        return "hybrid: semantic embeddings and keyword BM25" if self.vectors is not None else "keyword BM25"

    def keyword_scores(self, query: str) -> list[float]:
        q = set(terms(query))
        out = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            out.append(sum(self.idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + norm) for t in q if t in doc))
        return out

    def search(self, query: str, k: int = RESULT_PASSAGES) -> list[Hit]:
        if not self.passages:
            return []
        n = len(self.passages)
        keyword = self.keyword_scores(query)
        keyword_order = [i for i in sorted(range(n), key=lambda i: -keyword[i]) if keyword[i] > 0]
        similarity: list[float] | None = None
        meaning_order: list[int] = []
        if self.vectors is not None and self.embedder is not None:
            similarity = (self.vectors @ self.embedder.query(query)).tolist()
            meaning_order = [i for i in sorted(range(n), key=lambda i: -similarity[i]) if similarity[i] >= MIN_SIMILARITY]
            keyword_order = [i for i in keyword_order if similarity[i] >= MIN_SIMILARITY]
        keyword_rank = {i: r for r, i in enumerate(keyword_order, 1)}
        meaning_rank = {i: r for r, i in enumerate(meaning_order, 1)}
        fused = {i: sum(1 / (RRF_K + ranks[i]) for ranks in (keyword_rank, meaning_rank) if i in ranks) for i in {*keyword_rank, *meaning_rank}}
        hits: list[Hit] = []
        per_post: Counter[str] = Counter()
        for i in sorted(fused, key=lambda i: -fused[i]):
            passage = self.passages[i]
            if per_post[passage.title] >= PASSAGES_PER_POST:
                continue
            per_post[passage.title] += 1
            hits.append(Hit(passage, keyword_rank.get(i), meaning_rank.get(i), similarity[i] if similarity else None))
            if len(hits) == k:
                break
        return hits


def search_result(library: Library, query: str) -> str:
    """What the RAG tool gives the model: a few passages, never whole posts, and how to use them."""
    if not library.passages:
        return "No published writing is loaded. Say you cannot search the writing now, and answer from the profile notes."
    hits = library.search(query)
    if not hits:
        return (
            f"RAG search ({library.method}) over {len(library.posts)} essays found no passage about this question. "
            "Say the writing does not cover it. Do not guess the view."
        )
    found = "\n\n".join(f'From the essay titled "{h.passage.title}":\n{h.passage.text}' for h in hits)
    return (
        f"RAG search ({library.method}) retrieved {len(hits)} passages from {len(library.passages)} passages "
        f"in {len(library.posts)} essays.\n\n{found}\n\n"
        "Answer in two or three spoken sentences, in your own words, as the view of the person who wrote it. "
        "Answer only from these passages. Do not read them aloud."
    )


def about_line(library: Library, owner: str) -> str:
    """What the profile prompt says about the writing. Its size does not grow with the number of posts."""
    if not library.posts:
        return ""
    described = f' The feed describes them as: "{library.writing.about}"' if library.writing.about else ""
    return f"{owner} has published {len(library.posts)} short essays on software engineering.{described}"


@lru_cache(maxsize=1)
def default_library() -> Library:
    """The writing that the server uses, read and indexed once per process."""
    from ai_call_center.config import get_settings

    settings = get_settings()
    writing = load_writing(settings.writing_feed_url, settings.writing_cache_path)
    embedder: Embedder | None = None
    if writing.posts and settings.writing_embedding_model:
        try:
            embedder = FastEmbedder(settings.writing_embedding_model, settings.writing_model_dir)
        except Exception as exc:  # noqa: BLE001 - keyword search still works without the model
            log.warning("writing: no embedding model (%s); keyword search only", exc)
    return Library(writing, embedder=embedder, vector_cache=settings.writing_cache_path.with_name("writing_vectors.npz"))
