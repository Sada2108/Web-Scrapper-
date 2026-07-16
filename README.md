# PCB Design Research Scraper

A Firecrawl-powered scraper + Streamlit frontend that turns a natural-language
circuit/PCB design prompt into
a research corpus: relevant application notes, datasheets, reference designs,
schematics/images, and extracted text — ready to feed into a downstream
auto-design pipeline.

## How it works

```
prompt  ──▶  generate_search_queries()      heuristic EE keyword extraction
        ──▶  Firecrawl /search              finds candidate pages per query
        ──▶  Firecrawl /scrape              pulls markdown + html per page
        ──▶  extract_interleaved_content()  walks the page top-to-bottom,
                                             scores each block (paragraph,
                                             heading, table, inline image)
                                             for relevance, and keeps the
                                             relevant ones IN ORIGINAL ORDER
        ──▶  ResearchCorpus                 JSON-serializable result set
```

Text and images are **not** scraped as two separate passes. Each source's
content comes back as one interleaved markdown block — text, then its
related image, then more text, then its image — exactly as they appeared on
the page, so an image never loses the paragraph that was actually describing
it. The Streamlit UI and the exported Markdown report both render this
directly (Streamlit's `st.markdown()` shows `![]()` images inline), giving
you one continuous read per source instead of a text tab and a separate
image-gallery tab.

This is a **research/context-gathering** stage, not a schematic generator —
it does not itself produce a netlist or PCB layout. It's meant to sit in
front of whatever generation/LLM step you use to actually design the circuit,
giving it grounded reference material (real chopper-amp app notes, real
electrometer guarding techniques, etc.) instead of hallucinated specs.

## Setup

```bash
pip install -r requirements.txt
export FIRECRAWL_API_KEY="fc-your-key-here"   # from firecrawl.dev, required
export GROK_API_KEY="xai-your-key-here"       # from console.x.ai, optional
streamlit run app.py
```

You can also paste either key directly into the sidebar in the app instead
of using environment variables.

**Firecrawl key is required** — it's what does the actual searching/scraping.
**Grok key is optional** — without it, the tool still scrapes and produces
the full interleaved text+image report; you just won't get the "🤖 AI
Summary" section at the top. With it, each run sends the scraped sources to
xAI's Grok API (`grok-4.3`) for a short synthesis: overall summary, key
design considerations, recommended approach, and takeaways.

## Files

- `scraper.py` — core engine (query generation, search, scrape, interleaved
  text+image extraction, optional Grok AI summary, JSON/Markdown report
  export). Usable standalone from a script or notebook, independent of
  Streamlit.
- `app.py` — Streamlit UI: prompt box, example prompts, progress bar, a
  single "Research Report" tab (AI summary + text/images interleaved per
  source) and
  an "Export" tab (Markdown report / JSON).
- `requirements.txt`

## Using it standalone (no UI)

```python
from scraper import FirecrawlResearcher

researcher = FirecrawlResearcher(api_key="fc-...")
corpus = researcher.run_pipeline(
    prompt="Design a picoammeter capable of measuring currents from 10 fA to 10 µA...",
    max_queries=6,
    results_per_query=4,
    max_sources_to_scrape=12,
)

for source in corpus.sources:
    print(source.title, source.url, len(source.images), "images")
```

Every run is also cached to `./cache/corpus_<hash>_<timestamp>.json` so you
don't lose results between sessions.

## Notes / things to harden before production

- **Query planning is currently heuristic** (regex keyword matching against
  an EE vocabulary list). For richer, more accurate query expansion, swap
  `generate_search_queries()` for a call to an LLM that turns the prompt into
  targeted search queries per sub-topic (op-amp topology, battery/regulator
  specs, EMI/guarding techniques, etc.).
- **Image relevance scoring is a simple keyword heuristic** (alt text/URL
  matching against words like "schematic", "circuit", "pinout"). For higher
  precision, consider a vision-model pass to actually classify whether an
  image is a schematic vs. a stock photo before showing it.
- **Rate limits / cost**: each search + scrape call consumes Firecrawl
  credits. Tune `max_queries` / `results_per_query` / `max_sources_to_scrape`
  in the sidebar to control cost.
- **PDF datasheets**: many of the best sources (TI/ADI app notes) are PDFs.
  Firecrawl's `scrape()` can parse PDFs directly (`formats=["markdown"]`
  works on PDF URLs too) — no extra code needed, but large PDFs may take
  longer to process.
