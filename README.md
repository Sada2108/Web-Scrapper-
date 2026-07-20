# Web-Scrapper

A Firecrawl-powered research scraper for electronics/PCB design. Enter a circuit design prompt and it pulls relevant schematics, datasheets, application notes, and reference designs from the web.

## Pipeline

```
prompt  ──▶  generate_search_queries()     keyword extraction (part numbers,
                                            circuit types, EE vocabulary)
        ──▶  Firecrawl /search             finds candidate pages per query
        ──▶  scrape_source()               Firecrawl /scrape with per-domain
                                            strategies (SE comment expansion,
                                            DigiKey fallback chain, etc.)
        ──▶  extract_interleaved_content()  keeps relevant blocks in original
                                            reading order (text + images together)
        ──▶  ResearchCorpus                 JSON-serializable result set
```

Text and images are extracted **interleaved** (not two separate passes) — each image stays attached to the paragraph that was actually describing it on the source page. The Streamlit UI and Markdown export both preserve this layout.

## What works

| Source | Status |
|---|---|
| **Stack Exchange** (electronics, stackoverflow, etc.) | ✅ Comments expanded via ClickAction+WaitAction |
| **TI.com** (product pages, datasheets) | ✅ Server-rendered HTML, no JS issues |
| **Wikipedia**, **AllAboutCircuits**, manufacturer sites | ✅ Static/light JS pages |
| **DigiKey** (parametric table, pricing, stock) | ❌ Blocked — see [Known limitations](#known-limitations) |

## Setup

```bash
git clone https://github.com/Sada2108/Web-Scrapper-.git
cd Web-Scrapper-

cp .env.example .env
# Edit .env: set FIRECRAWL_API_KEY (required), GROK_API_KEY (optional)

pip install -r requirements.txt
```

### Run via Streamlit UI

```bash
streamlit run app.py
```

Paste your API keys in the sidebar or set `FIRECRAWL_API_KEY` / `GROK_API_KEY` env vars.

### Run standalone (no UI)

```python
from scraper import FirecrawlResearcher

researcher = FirecrawlResearcher(api_key="fc-...")
corpus = researcher.run_pipeline(
    prompt="Design a picoammeter capable of measuring currents from 10 fA to 10 µA...",
    max_queries=6,
    results_per_query=4,
    max_sources_to_scrape=12,
)
```

### Run tests

```bash
FIRECRAWL_API_KEY="fc-..." python3 test_scraper.py
```

Tests 4 suites (Stack Exchange, DigiKey, TI.com, Wikipedia) — exits 0 on pass, 1 on fail.

## Files

| File | Purpose |
|---|---|
| `scraper.py` | Core engine: query generation, search, scrape, relevance scoring, interleaved extraction, Grok summary, cache |
| `app.py` | Streamlit UI |
| `test_scraper.py` | Regression tests (12 assertions across 4 URL suites) |
| `check_grok.py` | Standalone Grok API key tester |
| `check.py` | Quick Firecrawl connectivity test |

## Known limitations

- **DigiKey parametric tables are not scrapable.** DigiKey's product pages are fully JS-rendered (React MuiSkeleton). Both Firecrawl Cloud (no Fire Engine for this domain) and self-hosted Firecrawl (Playwright) fail — DigiKey blocks automated access at the network level. The correct path is the [DigiKey Product Information v4 API](https://developer.digikey.com) (`api.digikey.com/products/v4/search/{partNumber}/productdetails`) which requires OAuth 2.0 credentials from a free DigiKey developer account. This integration is sketched but not yet implemented.
- **Query planning is heuristic** (regex keyword matching against an EE vocabulary). An LLM-based query planner would produce more targeted searches.
- **Image relevance scoring** is also keyword-based. A vision model pass would improve precision.
