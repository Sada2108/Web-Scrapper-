"""
scraper.py
----------
Core engine that turns a natural-language PCB/circuit design prompt into a
research corpus (text, schematics, images, source links) using Firecrawl.

Pipeline:
  1. prompt  -> generate_search_queries()   (keyword/topic extraction)
  2. queries -> search_sources()            (Firecrawl /search)
  3. sources -> scrape_source()             (Firecrawl /scrape, markdown+html)
  4. html    -> extract_images()            (filter for schematic/circuit imgs)
  5. everything -> run_pipeline()           (orchestrates + caches to JSON)

Requires: pip install firecrawl-py python-dotenv
Env var:  FIRECRAWL_API_KEY  (or pass api_key= explicitly)
"""

from __future__ import annotations

import json
import os
import re
import time
import hashlib
import requests
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse

from firecrawl import Firecrawl

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

CACHE_DIR = Path(os.environ.get("PCB_SCRAPER_CACHE", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# xAI Grok API (OpenAI-compatible chat completions endpoint). Used for the
# optional "AI Summary" pass over the scraped corpus -- everything else in
# this file works with zero LLM calls, this is purely additive.
GROK_API_BASE = "https://api.x.ai/v1"
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-4.3")

# Words/phrases that make a query more likely to return schematic-rich,
# electronics-engineering sources rather than generic marketing pages.
SOURCE_HINT_SUFFIXES = [
    "application note",
    "datasheet",
    "reference design",
    "schematic",
    "circuit design",
    "pinout",
    "typical application circuit",
    "components needed",
    "how to use",
    "circuit diagram",
]

# Trusted-ish EE domains we bias toward when present in results (not a hard
# filter -- just used for ranking).
PREFERRED_DOMAINS = [
    "ti.com", "analog.com", "onsemi.com", "microchip.com", "st.com",
    "renesas.com", "maximintegrated.com", "nxp.com", "allaboutcircuits.com",
    "electronics-tutorials.ws", "eevblog.com", "electronicdesign.com",
    "circuitdigest.com", "edn.com",
]

# A lightweight EE vocabulary used to pull technical terms out of the prompt
# when we don't want to (or can't) call an LLM to do query planning.
EE_KEYWORD_PATTERNS = [
    r"chopper[- ]stabilized amplifier", r"auto[- ]zero amplifier",
    r"electrometer[- ]grade amplifier", r"transimpedance amplifier",
    r"nanovoltmeter", r"picoammeter", r"femtoammeter",
    r"Kelvin (?:input )?connection", r"guard(?:ing)? (?:ring|technique)",
    r"EMI filtering", r"auto[- ]ranging", r"feedback resistor",
    r"offset drift", r"input bias current", r"low[- ]IQ regulator",
    r"Li-?ion battery", r"op[- ]amp", r"instrumentation amplifier",
    r"ADC", r"DAC", r"low noise amplifier", r"shunt resistor",
    r"current sense", r"voltage reference", r"PCB layout", r"ground plane",
    # Generic circuit types — common request shapes that should hit a real
    # pattern before ever reaching the naive word-split fallback.
    r"audio amplifier", r"power supply", r"voltage regulator",
    r"power amplifier", r"oscillator", r"filter circuit",
    r"LED driver", r"motor driver", r"battery charger",
    r"class [A-D] amplifier", r"preamp(?:lifier)?",
    r"audio circuit", r"amplifier circuit",
]

# Generic part-number regex — matches IC part numbers like LM386, TL071,
# OPA2340, MAX232, NE555 etc.  Run against the ORIGINAL-CASE prompt
# (part numbers are case-sensitive, unlike the EE_KEYWORD_PATTERNS which
# are searched case-insensitively on a lowercased prompt).
_PART_NUMBER_RE = re.compile(r"\b([A-Z]{2,5})[\s\-]?(\d{2,5}[A-Z]?)\b")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class ScrapedImage:
    url: str
    alt: str = ""
    relevance_score: int = 0


@dataclass
class Source:
    url: str
    title: str = ""
    query: str = ""
    markdown: str = ""
    images: List[ScrapedImage] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class ResearchCorpus:
    prompt: str
    queries: List[str]
    sources: List[Source]
    generated_at: float = field(default_factory=time.time)
    summary: Optional[str] = None
    summary_error: Optional[str] = None

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "queries": self.queries,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "summary_error": self.summary_error,
            "sources": [s.to_dict() for s in self.sources],
        }


# --------------------------------------------------------------------------
# Step 1: prompt -> search queries
# --------------------------------------------------------------------------

def generate_search_queries(prompt: str, max_queries: int = 6) -> List[str]:
    """
    Extract technical sub-topics from a free-text design prompt and turn
    them into targeted search queries. This is a heuristic keyword matcher
    (no external LLM call required); swap in an LLM-based planner later if
    you want richer query expansion.
    """
    # --- 1. Match known EE keyword patterns (case-insensitive) ---
    pattern_matches = []
    lower_prompt = prompt.lower()
    for pattern in EE_KEYWORD_PATTERNS:
        m = re.search(pattern, lower_prompt, flags=re.IGNORECASE)
        if m:
            term = m.group(0).strip()
            if term.lower() not in [f.lower() for f in pattern_matches]:
                pattern_matches.append(term)

    # --- 2. Extract IC part numbers (case-sensitive, original prompt) ---
    # Part numbers (LM386, TL071, OPA2340, ...) are strong, unambiguous
    # signals and are always prioritized over generic English words.
    # Tolerate a space or hyphen between the letter prefix and digit suffix
    # (e.g. "LM 386", "LM-386") and normalize by joining without separator.
    part_numbers = []
    for m in _PART_NUMBER_RE.finditer(prompt):
        normalized = m.group(1) + m.group(2)
        if normalized not in part_numbers:
            part_numbers.append(normalized)

    # --- 3. Fallback: naive word-split when neither pattern matched ---
    # Allow alphanumeric characters so part numbers survive the split
    # (the old regex `[A-Za-z][A-Za-z\-]{3,}` silently dropped anything
    # containing digits like "LM386").  Apply STOPWORDS so filler words
    # like "using", "with", "from" never become search queries.
    fallback_words = []
    if not pattern_matches and not part_numbers:
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", prompt)
        fallback_words = [
            w for w in dict.fromkeys(words)
            if w.lower() not in STOPWORDS
        ]

    # --- 4. Merge: part numbers first, then pattern matches, then fallback ---
    found = part_numbers + pattern_matches + fallback_words
    seen = set()
    unique = []
    for term in found:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            unique.append(term)
    found = unique

    # --- 5. Build queries: spread suffixes across the full list ---
    # When max_queries < len(SOURCE_HINT_SUFFIXES), pick suffixes spread
    # evenly across the entire list instead of taking the first N.  E.g.
    # max_queries=6 out of 10 suffixes gives indices [0,1,3,5,6,8].
    n_suffixes = len(SOURCE_HINT_SUFFIXES)
    if max_queries <= n_suffixes:
        step = n_suffixes / max_queries
        suffix_indices = [int(i * step) for i in range(max_queries)]
    else:
        suffix_indices = list(range(n_suffixes))

    queries = []
    if found:
        total_terms = len(found)
        # Per-term quotas: strongest term gets ceil, rest get floor.
        quotas = []
        for ti in range(total_terms):
            base = max_queries // total_terms
            extra = 1 if ti < (max_queries - base * total_terms) else 0
            quotas.append(base + extra)
        term_counts = [0] * total_terms
        si = 0  # index into suffix_indices
        while len(queries) < max_queries:
            added = False
            for ti in range(total_terms):
                if term_counts[ti] < quotas[ti] and si < len(suffix_indices):
                    term = found[ti]
                    suffix = SOURCE_HINT_SUFFIXES[suffix_indices[si]]
                    queries.append(f"{term} {suffix}")
                    si += 1
                    term_counts[ti] += 1
                    added = True
                    if len(queries) >= max_queries:
                        break
            if not added:
                break

    if not queries:
        queries = [prompt[:80]]

    return queries


# --------------------------------------------------------------------------
# Step 2 & 3: search + scrape
# --------------------------------------------------------------------------

class FirecrawlResearcher:
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError(
                "No Firecrawl API key found. Set FIRECRAWL_API_KEY env var "
                "or pass api_key= explicitly."
            )
        self.client = Firecrawl(api_key=api_key)

    # -- search ------------------------------------------------------------
    def search_sources(self, query: str, limit: int = 5) -> List[Dict]:
        """Run a Firecrawl web search and return raw result dicts."""
        try:
            result = self.client.search(query, limit=limit)
        except Exception as e:
            return [{"url": None, "title": None, "error": str(e)}]

        # firecrawl-py v2 returns an object with `.web` (list) in most
        # versions; fall back to treating result as a plain list/dict.
        items = getattr(result, "web", None)
        if items is None:
            items = result if isinstance(result, list) else result.get("web", [])

        out = []
        for item in items[:limit]:
            url = getattr(item, "url", None) or item.get("url")
            title = getattr(item, "title", None) or item.get("title", "")
            out.append({"url": url, "title": title})
        return out

    # -- scrape --------------------------------------------------------------
    def scrape_source(self, url: str, query: str = "", prompt: str = "") -> Source:
        """Scrape a single URL for markdown + html, then pull out images."""
        try:
            doc = self.client.scrape(
                url,
                formats=["markdown", "html", "links"],
                # Deterministic HTML-level filter: strips nav/header/footer/ads
                # before markdown conversion even happens.
                only_main_content=True,
                # Beta: extra LLM pass to mop up residual boilerplate
                # (cookie banners, share widgets, related-article lists) that
                # only_main_content can miss. Falls back gracefully if a given
                # firecrawl-py version doesn't support the kwarg yet.
                only_clean_content=True,
            )
        except TypeError:
            # older firecrawl-py without only_clean_content support
            try:
                doc = self.client.scrape(
                    url, formats=["markdown", "html", "links"], only_main_content=True
                )
            except Exception as e:
                return Source(url=url, query=query, error=str(e))
        except Exception as e:
            return Source(url=url, query=query, error=str(e))

        markdown = getattr(doc, "markdown", None) or (
            doc.get("markdown", "") if isinstance(doc, dict) else ""
        )
        html = getattr(doc, "html", None) or (
            doc.get("html", "") if isinstance(doc, dict) else ""
        )
        meta = getattr(doc, "metadata", None) or (
            doc.get("metadata", {}) if isinstance(doc, dict) else {}
        )
        title = ""
        if meta:
            title = getattr(meta, "title", None) or (
                meta.get("title", "") if isinstance(meta, dict) else ""
            )

        # Interleaved pass: text and images are extracted TOGETHER, in the
        # order they actually appear on the page, so an image never loses
        # the paragraph that gives it context.
        content, images = extract_interleaved_content(
            markdown or "", prompt, max_chars=12000
        )

        # Safety net: some pages embed images (lazy-loaded, CSS background,
        # <picture> tags) that Firecrawl's markdown conversion drops even
        # though they're present in the raw HTML. Sweep the HTML too and
        # append anything relevant that isn't already captured above, so a
        # real schematic never gets silently lost just because it didn't
        # survive the markdown conversion. Unlike the interleaved pass,
        # this sweep has no surrounding paragraph to judge context from --
        # so it requires an actual positive keyword signal (score > 0)
        # rather than just "not obviously junk", to avoid pulling in
        # unrelated page furniture that never had a chance to be filtered
        # by context in the first place.
        already_have = {img.url for img in images}
        html_images = [
            img for img in extract_images(html or "", "")
            if img.url not in already_have and img.relevance_score > 0
        ]
        if html_images:
            extra_block = "\n\n**📎 Additional images found on this page:**\n\n" + "\n\n".join(
                f"![{img.alt or 'image'}]({img.url})" for img in html_images[:15]
            )
            content = (content + extra_block) if content else extra_block.strip()
            images = images + html_images[:15]

        return Source(
            url=url,
            title=title or url,
            query=query,
            markdown=content,
            images=images,
        )

    # -- full pipeline -------------------------------------------------------
    def run_pipeline(
        self,
        prompt: str,
        max_queries: int = 6,
        results_per_query: int = 4,
        max_sources_to_scrape: int = 12,
        progress_cb=None,
        grok_api_key: Optional[str] = None,
    ) -> ResearchCorpus:
        """
        End-to-end: prompt -> queries -> search -> scrape -> corpus.
        progress_cb(stage: str, current: int, total: int) is called for
        UI progress bars (e.g. from Streamlit).
        """
        queries = generate_search_queries(prompt, max_queries=max_queries)

        candidates: List[Dict] = []
        for i, q in enumerate(queries):
            if progress_cb:
                progress_cb("searching", i + 1, len(queries))
            results = self.search_sources(q, limit=results_per_query)
            for r in results:
                if r.get("url"):
                    r["query"] = q
                    candidates.append(r)

        candidates = _dedupe_and_rank(candidates)[:max_sources_to_scrape]

        sources: List[Source] = []
        for i, c in enumerate(candidates):
            if progress_cb:
                progress_cb("scraping", i + 1, len(candidates))
            src = self.scrape_source(c["url"], query=c.get("query", ""), prompt=prompt)
            if not src.title:
                src.title = c.get("title", src.url)
            sources.append(src)

        corpus = ResearchCorpus(prompt=prompt, queries=queries, sources=sources)

        if grok_api_key:
            if progress_cb:
                progress_cb("summarizing", 1, 1)
            summarizer = GrokSummarizer(api_key=grok_api_key)
            result = summarizer.summarize_corpus(corpus)
            if result:
                corpus.summary = result.get("text")
                corpus.summary_error = result.get("error")

        _save_cache(corpus)
        return corpus


class GrokSummarizer:
    """
    Optional AI Summary pass using xAI's Grok API (OpenAI-compatible
    /chat/completions endpoint). This never blocks or breaks the rest of
    the pipeline -- if there's no key, or the call fails, run_pipeline()
    just leaves corpus.summary as None and everything else still works.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROK_API_KEY")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def summarize_corpus(self, corpus: "ResearchCorpus", max_sources: int = 8) -> Optional[Dict]:
        if not self.enabled:
            return None
        ok_sources = [s for s in corpus.sources if not s.error][:max_sources]
        if not ok_sources:
            return None

        digest = "\n\n".join(
            f"### {s.title} ({s.url})\n{s.markdown[:2500]}" for s in ok_sources
        )
        system_prompt = (
            "You are an electronics design research assistant. Given scraped "
            "application notes, datasheets, and reference designs relevant to "
            "a PCB/circuit design prompt, produce a concise, technically "
            "specific digest an EE could act on. Respond ONLY in this exact "
            "markdown structure, no preamble, no extra commentary:\n\n"
            "## 🤖 AI Summary\n<3-5 sentence synthesis across all sources>\n\n"
            "## 🎯 Key Design Considerations\n<4-6 bullet points, specific "
            "and technical -- topologies, component choices, tradeoffs>\n\n"
            "## 📐 Recommended Approach\n<2-4 bullet points>\n\n"
            "## 💡 Key Takeaways\n<3-5 bullet points>"
        )
        user_prompt = (
            f"Design prompt:\n{corpus.prompt}\n\n"
            f"Scraped source material:\n{digest[:18000]}"
        )
        try:
            resp = requests.post(
                f"{GROK_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROK_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1200,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return {"text": text}
        except Exception as e:
            return {"error": f"Grok summary failed: {e}"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

IMG_TAG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)
ALT_RE = re.compile(r'alt=["\']([^"\']*)["\']', re.IGNORECASE)
MD_IMG_RE = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)')

RELEVANCE_KEYWORDS = [
    "schematic", "circuit", "diagram", "block diagram", "pinout",
    "waveform", "topology", "layout", "pcb", "wiring",
]

# Common filler words that happen to be 4+ letters (the threshold used to
# pull "significant" words out of the prompt) but carry no real technical
# signal -- without excluding these, a totally unrelated sentence that
# merely contains "with" or "from" can score as if it matched the prompt.
STOPWORDS = {
    "with", "from", "that", "this", "these", "those", "when", "where",
    "have", "will", "your", "into", "only", "also", "such", "than",
    "then", "them", "they", "were", "been", "being", "each", "some",
    "more", "most", "other", "which", "while", "about", "after",
    "before", "over", "under", "between", "using", "used", "used.",
    "here", "there", "what", "very", "just", "like", "make", "made",
}

# Filenames/alt-text containing these almost never point to an actual
# schematic/circuit image, even when they carry no positive keyword either
# -- product marketing renders, conference/booth photos, author headshots,
# "related articles" carousel thumbnails, ads, and social-share cards.
JUNK_IMAGE_KEYWORDS = [
    "logo", "icon", "avatar", "sprite", "banner", "hero", "promo",
    "campaign", "press-release", "press_release", "booth", "conference",
    "event", "team-photo", "team_photo", "headshot", "portrait", "author",
    "staff", "thumbnail", "thumb", "og-image", "og_image", "social-share",
    "social_share", "card-image", "card_image", "related-article",
    "related_article", "recommend", "sponsor", "advert", "stock-photo",
    "stockphoto",
]

# Line-level noise: cookie notices, nav/share/subscribe cruft that
# only_main_content sometimes lets through anyway.
NOISE_LINE_PATTERNS = [
    r"cookie", r"we use cookies", r"accept all cookies", r"privacy policy",
    r"subscribe to our newsletter", r"sign in", r"log in", r"create an account",
    r"advertisement", r"sponsored", r"related articles", r"you may also like",
    r"share this", r"follow us on", r"skip to (main )?content", r"back to top",
    r"all rights reserved", r"terms of (use|service)",
]
_NOISE_LINE_RE = re.compile("|".join(NOISE_LINE_PATTERNS), re.IGNORECASE)
_MD_LINK_ONLY_RE = re.compile(r"^\s*[-*]?\s*\[([^\]]+)\]\(([^)]+)\)\s*$")
_MD_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")


def _is_nav_row(stripped_line: str) -> bool:
    """A line that's nothing but markdown links + separators (|, ,, ·) is a
    nav/breadcrumb row, e.g. '[Home](/) | [About](/about) | [Contact](/c)'."""
    if not _MD_LINK_RE.search(stripped_line):
        return False
    remainder = _MD_LINK_RE.sub("", stripped_line)
    return not re.sub(r"[|,·\-\s]", "", remainder)

# A number followed by an EE-relevant unit is a strong "this paragraph has
# real technical content" signal (e.g. "100 nV", "2 mA", "10 kOhm").
_UNIT_NUMBER_RE = re.compile(
    r"\d+(\.\d+)?\s*(m?v|m?a|u?a|n?a|p?a|f?a|ohm|hz|khz|mhz|ghz|db|nf|pf|uf|"
    r"bit|sps|ppm|ppb|degc|°c)\b",
    re.IGNORECASE,
)


def _split_blocks(markdown: str) -> List[str]:
    """Split markdown into blocks on blank lines, preserving original order.
    A 'block' is a paragraph, heading, table row-group, or inline image —
    whatever markdown naturally groups between blank lines."""
    raw_blocks = re.split(r"\n\s*\n", markdown or "")
    return [b.strip() for b in raw_blocks if b.strip()]


def _block_images(block: str) -> List["ScrapedImage"]:
    imgs = []
    for m in MD_IMG_RE.finditer(block):
        alt, url = m.group(1), m.group(2)
        imgs.append(ScrapedImage(url=url, alt=alt, relevance_score=_score_image(url, alt)))
    return imgs


def _block_text_score(block: str, prompt_terms, prompt_words) -> int:
    lower = block.lower()
    score = 0
    for term in prompt_terms:
        if term in lower:
            score += 3
    score += sum(1 for w in prompt_words if w in lower)
    score += sum(1 for kw in RELEVANCE_KEYWORDS if kw in lower)
    if _UNIT_NUMBER_RE.search(lower):
        score += 2
    if block.startswith("#") or "|" in block:
        score += 1
    return score


def extract_interleaved_content(
    markdown: str, prompt: str, max_chars: int = 9000
):
    """
    Walk the scraped markdown top-to-bottom and keep the blocks (paragraphs,
    headings, tables, AND inline images) relevant to the design prompt --
    in their ORIGINAL reading order.

    This is the key difference from scraping text and images separately:
    an image stays attached to whatever paragraph was actually talking
    about it, instead of getting dumped into a disconnected gallery. It's
    also why a relevant image whose *own* alt text is weak (e.g. a
    schematic with alt="Figure 3") still gets kept -- it inherits
    relevance from the surrounding paragraph via the neighbor-context step
    below, instead of being scored purely in isolation.

    Returns (interleaved_markdown, images_kept) so callers get one ready
    to render/save string plus a flat image list for stats/galleries.
    """
    cleaned = _clean_boilerplate(markdown)
    blocks = _split_blocks(cleaned)
    if not blocks:
        return "", []

    prompt_lower = (prompt or "").lower()
    prompt_terms = set()
    for pattern in EE_KEYWORD_PATTERNS:
        m = re.search(pattern, prompt_lower, flags=re.IGNORECASE)
        if m:
            prompt_terms.add(m.group(0).lower())
    prompt_words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", prompt or "")} - STOPWORDS

    # classify + score every block, keeping original index for reordering
    scored = []  # (idx, kind, score, block, images_in_block)
    for idx, block in enumerate(blocks):
        imgs = _block_images(block)
        text_without_imgs = MD_IMG_RE.sub("", block).strip()
        is_image_block = bool(imgs) and len(text_without_imgs) < 20
        if is_image_block:
            score = max((i.relevance_score for i in imgs), default=0)
            scored.append((idx, "image", score, block, imgs))
        else:
            if len(block) < 40:
                continue  # stray menu items / bare links, not real content
            score = _block_text_score(block, prompt_terms, prompt_words)
            scored.append((idx, "text", score, block, imgs))

    if not scored:
        return cleaned[:max_chars], []

    # Text blocks: keep if they scored positively against the prompt.
    #
    # Image blocks are trickier: an image's own alt text is often blank or
    # generic ("Figure 3", "img_042.png") even when it's a real schematic,
    # so requiring a positive keyword match on the image alone drops real
    # circuit diagrams. But being too permissive the other way (keep any
    # image that isn't *explicitly* junk, anywhere on the page) lets
    # unrelated stuff through too -- product marketing renders, "related
    # articles" carousel thumbnails, event photos -- none of which happen
    # to contain a junk keyword either.
    #
    # The fix is context: a neutral-scoring image is only kept if it's
    # actually sitting near text that's genuinely relevant to the prompt.
    # An image with its OWN positive signal (alt/filename says
    # "schematic") is always kept regardless of neighbors. An image that's
    # explicitly junk (logo, banner, headshot, ...) is never kept.
    def _nearby_text_score(pos: int, window: int = 2) -> int:
        best = 0
        for offset in range(1, window + 1):
            for neighbor_pos in (pos - offset, pos + offset):
                if 0 <= neighbor_pos < len(scored):
                    n_idx, n_kind, n_score, n_block, n_imgs = scored[neighbor_pos]
                    if n_kind == "text":
                        best = max(best, n_score)
        return best

    keep = set()
    for pos, (idx, kind, score, block, imgs) in enumerate(scored):
        if kind == "text":
            if score > 0:
                keep.add(idx)
        else:  # image
            if score > 0:
                keep.add(idx)  # own alt/filename gives a real signal
            elif score == 0 and _nearby_text_score(pos) > 0:
                keep.add(idx)  # neutral, but sits near relevant content
            # score < 0 (explicit junk) is never kept, regardless of context

    # nothing matched the prompt at all -- fall back to the longest blocks
    # rather than returning an empty source
    if not keep:
        longest = sorted(scored, key=lambda s: len(s[3]), reverse=True)[:10]
        keep = {s[0] for s in longest}

    # give every kept image one block of surrounding TEXT as context (its
    # caption or lead-in sentence) so it never shows up floating with no
    # explanation around it. Deliberately one-directional: a relevant
    # paragraph should NOT pull in a neighboring image just because it's
    # adjacent (that's how logos/icons next to real content used to sneak
    # in) -- only images pull in text, never the reverse.
    for pos, (idx, kind, score, block, imgs) in enumerate(scored):
        if kind == "image" and idx in keep:
            for neighbor_pos in (pos - 1, pos + 1):
                if 0 <= neighbor_pos < len(scored):
                    n_idx, n_kind, n_score, n_block, n_imgs = scored[neighbor_pos]
                    if n_kind == "text":
                        keep.add(n_idx)

    # Respect the character budget for TEXT only -- images are exempt from
    # the budget entirely (capped at a generous max_images instead) so a
    # long page never causes a real schematic to get trimmed out purely
    # because text elsewhere ate the budget first.
    by_score = sorted(scored, key=lambda s: s[2], reverse=True)
    budget_keep, total = set(), 0
    image_count = 0
    max_images = 20
    for idx, kind, score, block, imgs in by_score:
        if idx not in keep:
            continue
        if kind == "image":
            if image_count >= max_images:
                continue
            budget_keep.add(idx)
            image_count += 1
            continue
        if total + len(block) > max_chars and budget_keep:
            continue
        budget_keep.add(idx)
        total += len(block)

    final_indices = sorted(budget_keep)
    kept_blocks = [blocks[i] for i in final_indices]
    kept_images = [
        img
        for idx, kind, score, block, imgs in scored
        if idx in budget_keep and kind == "image"
        for img in imgs
    ]

    return "\n\n".join(kept_blocks), kept_images


def extract_images(html: str, markdown: str = "") -> List[ScrapedImage]:
    found: Dict[str, ScrapedImage] = {}

    for m in IMG_TAG_RE.finditer(html or ""):
        src = m.group(0)
        url = m.group(1)
        alt_match = ALT_RE.search(src)
        alt = alt_match.group(1) if alt_match else ""
        found[url] = ScrapedImage(url=url, alt=alt, relevance_score=_score_image(url, alt))

    for m in MD_IMG_RE.finditer(markdown or ""):
        alt, url = m.group(1), m.group(2)
        if url not in found:
            found[url] = ScrapedImage(url=url, alt=alt, relevance_score=_score_image(url, alt))

    images = list(found.values())
    images.sort(key=lambda i: i.relevance_score, reverse=True)
    return images


def _score_image(url: str, alt: str) -> int:
    text = f"{url} {alt}".lower()
    score = 0
    for kw in RELEVANCE_KEYWORDS:
        if kw in text:
            score += 2
    # Penalize obvious non-content images
    if any(kw in text for kw in JUNK_IMAGE_KEYWORDS):
        score -= 3
    # Small tracking pixels / svg sprites are rarely useful
    if url.lower().endswith((".svg",)) and "schematic" not in text:
        score -= 1
    return score


def _clean_boilerplate(markdown: str) -> str:
    """
    Strip cookie notices, nav/share/subscribe cruft, and runs of bare
    menu-style links from scraped markdown. Cheap, deterministic pass that
    runs on top of Firecrawl's own only_main_content filtering.
    """
    lines = (markdown or "").splitlines()
    cleaned = []
    consecutive_link_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            consecutive_link_lines = 0
            continue

        if _NOISE_LINE_RE.search(stripped):
            continue

        if _is_nav_row(stripped):
            continue

        if _MD_LINK_ONLY_RE.match(stripped):
            consecutive_link_lines += 1
            # 3+ consecutive bare-link lines is almost always a nav/menu block
            if consecutive_link_lines >= 3:
                continue
        else:
            consecutive_link_lines = 0

        cleaned.append(line)

    return "\n".join(cleaned)


def extract_relevant_text(markdown: str, prompt: str, max_chars: int = 6000) -> str:
    """
    Rank paragraphs of scraped markdown by relevance to the design prompt
    (same spirit as _score_image for images) instead of just truncating
    from the top of the page, which usually grabs boilerplate.
    """
    cleaned = _clean_boilerplate(markdown)
    paragraphs = [p for p in re.split(r"\n\s*\n", cleaned)]

    prompt_lower = (prompt or "").lower()
    prompt_terms = set()
    for pattern in EE_KEYWORD_PATTERNS:
        m = re.search(pattern, prompt_lower, flags=re.IGNORECASE)
        if m:
            prompt_terms.add(m.group(0).lower())
    prompt_words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", prompt or "")} - STOPWORDS

    scored = []
    for idx, para in enumerate(paragraphs):
        text = para.strip()
        if len(text) < 40:
            # too short to be real content (stray menu items, single links)
            continue
        lower = text.lower()
        score = 0
        for term in prompt_terms:
            if term in lower:
                score += 3
        score += sum(1 for w in prompt_words if w in lower)
        score += sum(1 for kw in RELEVANCE_KEYWORDS if kw in lower)
        if _UNIT_NUMBER_RE.search(lower):
            score += 2
        # headings and tables often carry key specs even if short on keywords
        if text.startswith("#") or "|" in text:
            score += 1
        scored.append((idx, score, text))

    if not scored:
        return cleaned[:max_chars]

    kept = [s for s in scored if s[1] > 0]
    if not kept:
        # nothing matched keywords -- fall back to the longest paragraphs
        # rather than dropping the source's text entirely
        kept = sorted(scored, key=lambda s: len(s[2]), reverse=True)[:10]

    kept_by_score = sorted(kept, key=lambda s: s[1], reverse=True)
    selected, total = [], 0
    for idx, score, text in kept_by_score:
        if total >= max_chars:
            break
        selected.append((idx, text))
        total += len(text)

    # restore original reading order for coherence
    selected.sort(key=lambda s: s[0])
    return "\n\n".join(t for _, t in selected)


def _dedupe_and_rank(candidates: List[Dict]) -> List[Dict]:
    seen = set()
    deduped = []
    for c in candidates:
        url = c.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(c)

    def rank(c):
        domain = urlparse(c["url"]).netloc.replace("www.", "")
        preferred = any(pd in domain for pd in PREFERRED_DOMAINS)
        return (0 if preferred else 1, domain)

    deduped.sort(key=rank)
    return deduped


def _save_cache(corpus: ResearchCorpus) -> Path:
    key = hashlib.sha256(corpus.prompt.encode("utf-8")).hexdigest()[:16]
    path = CACHE_DIR / f"corpus_{key}_{int(corpus.generated_at)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def corpus_to_markdown(corpus: ResearchCorpus) -> str:
    """
    Flatten a corpus into a single, clean flowing markdown report --
    text and its related images interleaved per source (no separate
    galleries), one document you can read top to bottom.
    """
    ok_sources = [s for s in corpus.sources if not s.error]
    failed_sources = [s for s in corpus.sources if s.error]
    total_images = sum(len(s.images) for s in ok_sources)
    title = corpus.prompt.strip().splitlines()[0][:90]

    lines = [
        f"# 🔩 PCB Design Research: {title}\n",
        f"**🧠 Full prompt:** {corpus.prompt}  ",
        f"**🔍 Search queries used ({len(corpus.queries)}):** {', '.join(corpus.queries)}  ",
        f"**📄 Sources scraped:** {len(ok_sources)}  ",
        f"**🖼️ Relevant images found:** {total_images}  ",
        f"**⚠️ Failed fetches:** {len(failed_sources)}  ",
        "\n---\n",
    ]

    if corpus.summary:
        lines.append(corpus.summary.strip() + "\n")
        lines.append("\n---\n")
    elif corpus.summary_error:
        lines.append(f"> ⚠️ AI summary unavailable: {corpus.summary_error}\n")
        lines.append("\n---\n")

    for i, s in enumerate(ok_sources, 1):
        lines.append(f"## {i}. {s.title}\n")
        lines.append(f"**🔗 URL:** [{s.url}]({s.url})  ")
        lines.append(f"**🔍 Matched query:** *{s.query}*  ")
        lines.append(f"**🖼️ Images in this section:** {len(s.images)}  \n")
        # Text and images are already interleaved in s.markdown, in the
        # order they appeared on the source page -- this is the "text,
        # then its related image, then text, then image" layout.
        lines.append(s.markdown.strip() if s.markdown.strip() else "*(no relevant content extracted)*")
        lines.append("\n---\n")

    if failed_sources:
        lines.append("## ⚠️ Failed to scrape\n")
        for s in failed_sources:
            lines.append(f"- {s.url} — {s.error}")

    return "\n".join(lines)
