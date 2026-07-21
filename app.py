"""
app.py
------
Streamlit frontend for the Firecrawl-powered PCB/circuit design research
scraper. Enter a design prompt (like a Flux-AI-style spec), and it will:

  1. Break the prompt into technical search queries
  2. Search the web with Firecrawl
  3. Scrape the top matching pages (text + images)
  4. Interleave the relevant text and its related images in original
     reading order (no separate "gallery" -- an image stays next to the
     paragraph that was actually describing it, so nothing loses context)
  5. Show you one clean, flowing research report per query
  6. Let you export everything as JSON / Markdown for downstream use
     (e.g. feeding into your PCB auto-design pipeline)

Run with:  streamlit run app.py
"""

import os
import re
import time
import streamlit as st

from scraper import (
    FirecrawlResearcher,
    generate_search_queries,
    corpus_to_markdown,
)

st.set_page_config(page_title="PCB Design Research Scraper", page_icon="🔩", layout="wide")

EXAMPLE_PROMPTS = {
    "Nanovoltmeter front end": (
        "Design a battery-operated nanovoltmeter front end capable of resolving "
        "signals below 100 nV while consuming less than 2 mA. Compare "
        "chopper-stabilized and auto-zero amplifier architectures. Include "
        "auto-ranging, Kelvin input connections, EMI filtering, and estimate "
        "offset drift, noise, and measurement accuracy."
    ),
    "Picoammeter": (
        "Design a picoammeter capable of measuring currents from 10 fA to 10 µA "
        "using an electrometer-grade amplifier. Operate from a single Li-ion "
        "battery with ultra-low-IQ regulators. Include guarding techniques, "
        "switchable feedback resistors, auto-ranging, and estimate input bias "
        "current, noise, and measurement resolution."
    ),
}

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "corpus" not in st.session_state:
    st.session_state.corpus = None
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = ""

# --------------------------------------------------------------------------
# Sidebar: settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Firecrawl API key",
        value=os.environ.get("FIRECRAWL_API_KEY", ""),
        type="password",
        help="Get one at firecrawl.dev. Falls back to FIRECRAWL_API_KEY env var.",
    )

    grok_api_key = st.text_input(
        "Grok (xAI) API key — optional",
        value=os.environ.get("GROK_API_KEY", ""),
        type="password",
        help=(
            "Get one at console.x.ai. Falls back to GROK_API_KEY env var. "
            "Powers the 'AI Summary' section at the top of the report. "
            "Leave blank to skip it — scraping and the interleaved "
            "text+image report work fine without it."
        ),
    )

    st.markdown("---")
    max_queries = st.slider("Search queries to generate", 2, 10, 10)
    results_per_query = st.slider("Results per query", 1, 10, 4)
    max_sources = st.slider("Max pages to scrape (total)", 3, 30, 12)

    st.markdown("---")
    st.caption(
        "This tool searches the public web for application notes, datasheets, "
        "and reference designs, then scrapes text and images relevant to your "
        "prompt. It does not itself generate a PCB layout — think of it as the "
        "research/context-gathering stage feeding into your design pipeline."
    )

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("🔩 PCB Design Research Scraper")
st.write(
    "Give it a circuit/PCB design prompt and it will pull relevant schematics, "
    "circuit images, and reference text from the web using Firecrawl."
)

cols = st.columns(len(EXAMPLE_PROMPTS))
for col, (label, text) in zip(cols, EXAMPLE_PROMPTS.items()):
    if col.button(f"📋 Use example: {label}", use_container_width=True):
        st.session_state.prompt_text = text

prompt = st.text_area(
    "Design prompt",
    value=st.session_state.prompt_text,
    height=160,
    placeholder="e.g. Design a battery-operated nanovoltmeter front end capable of resolving signals below 100 nV...",
)
st.session_state.prompt_text = prompt

with st.expander("🔍 Preview generated search queries"):
    if prompt.strip():
        st.write(generate_search_queries(prompt, max_queries=max_queries))
    else:
        st.caption("Enter a prompt above to preview the queries it will run.")

run = st.button("🚀 Run research scrape", type="primary", disabled=not prompt.strip())

# --------------------------------------------------------------------------
# Run pipeline
# --------------------------------------------------------------------------
if run:
    if not api_key:
        st.error("Please provide a Firecrawl API key in the sidebar.")
        st.stop()

    try:
        researcher = FirecrawlResearcher(api_key=api_key)
    except Exception as e:
        st.error(f"Could not initialize Firecrawl client: {e}")
        st.stop()

    progress_bar = st.progress(0, text="Starting...")
    status = st.empty()

    def progress_cb(stage, current, total):
        pct = int((current / max(total, 1)) * 100)
        pct = min(max(pct, 0), 100)
        labels = {
            "searching": "Searching the web",
            "scraping": "Scraping pages",
            "summarizing": "Generating AI summary",
        }
        progress_bar.progress(pct, text=f"{labels.get(stage, stage)} ({current}/{total})")

    start = time.time()
    try:
        corpus = researcher.run_pipeline(
            prompt=prompt,
            max_queries=max_queries,
            results_per_query=results_per_query,
            max_sources_to_scrape=max_sources,
            progress_cb=progress_cb,
            grok_api_key=grok_api_key or None,
        )
        st.session_state.corpus = corpus
        progress_bar.progress(100, text="Done")
        status.success(f"Finished in {time.time() - start:.1f}s — {len(corpus.sources)} pages scraped.")
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
corpus = st.session_state.corpus

if corpus:
    st.markdown("---")
    ok_sources = [s for s in corpus.sources if not s.error]
    failed_sources = [s for s in corpus.sources if s.error]
    total_images = sum(len(s.images) for s in ok_sources)

    m1, m2, m3 = st.columns(3)
    m1.metric("Pages scraped", len(ok_sources))
    m2.metric("Relevant images found", total_images)
    m3.metric("Failed fetches", len(failed_sources))

    tab_report, tab_export = st.tabs(["📊 Research Report", "⬇️ Export"])

    with tab_report:
        st.caption(
            f"🔍 Query: *{corpus.prompt[:100]}{'...' if len(corpus.prompt) > 100 else ''}*  "
            f"·  {len(ok_sources)} sources  ·  {total_images} images"
        )

        if corpus.summary:
            with st.container(border=True):
                st.markdown(corpus.summary)
            st.markdown("")
        elif corpus.summary_error:
            st.warning(corpus.summary_error)

        # One flowing document per source, text and its related image(s)
        # already interleaved in original reading order -- st.markdown
        # renders the inline ![]() image syntax right where it occurs, so
        # you see "text, then its image, then text, then its image" as a
        # single continuous read instead of split tabs.
        #
        # PDF-extracted figures are stored as local files, not URLs --
        # st.markdown can't serve arbitrary local paths, so we render
        # those with st.image() instead and strip the broken refs from
        # the markdown.
        for i, s in enumerate(ok_sources, 1):
            with st.container(border=True):
                st.markdown(f"### {i}. {s.title}")
                st.caption(f"🔗 [{s.url}]({s.url})  ·  matched query: *{s.query}*  ·  {len(s.images)} images")
                local_imgs = [img for img in s.images if not img.url.startswith("http")]
                if s.markdown.strip():
                    # Strip local-path image refs from markdown (they won't render)
                    if local_imgs:
                        clean = re.sub(
                            r'!\[([^\]]*)\]\((?!https?://)([^)]+)\)',
                            '', s.markdown,
                        )
                        # Also remove the "Figures extracted from PDF:" heading
                        # and surrounding blank lines left after stripping
                        clean = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean).strip()
                    else:
                        clean = s.markdown
                    st.markdown(clean, unsafe_allow_html=False)
                    for img in local_imgs:
                        from pathlib import Path
                        path = Path(img.url)
                        if path.exists():
                            st.image(str(path), caption=img.alt or None)
                else:
                    st.info("No relevant content extracted from this page for your prompt.")

        if failed_sources:
            with st.expander(f"⚠️ {len(failed_sources)} pages failed to scrape"):
                for s in failed_sources:
                    st.caption(f"❌ {s.url} — {s.error}")

    with tab_export:
        md_export = corpus_to_markdown(corpus)
        json_export = __import__("json").dumps(corpus.to_dict(), indent=2, ensure_ascii=False)

        st.markdown("**Preview of the exported report:**")
        with st.expander("Preview", expanded=False):
            st.markdown(md_export, unsafe_allow_html=False)

        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇️ Download as Markdown",
            data=md_export,
            file_name="pcb_research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        c2.download_button(
            "⬇️ Download as JSON",
            data=json_export,
            file_name="pcb_research_corpus.json",
            mime="application/json",
            use_container_width=True,
        )
        st.caption(
            "The Markdown export is the same interleaved text+image report shown "
            "in the Research Report tab, ready to read or hand to a downstream "
            "PCB auto-design / LLM pipeline. The JSON export gives you the "
            "structured per-source data (interleaved markdown + image list) if "
            "you want to process it programmatically instead."
        )
else:
    st.info("Enter a prompt and click **Run research scrape** to get started.")
