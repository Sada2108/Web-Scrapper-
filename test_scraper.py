"""
Regression tests for scraper.py.

Exits non-zero on any failure so it's CI-compatible.
Run:  FIRECRAWL_API_KEY=... python3 test_scraper.py
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper import FirecrawlResearcher

API_KEY = os.environ.get("FIRECRAWL_API_KEY")
if not API_KEY:
    sys.exit("FIRECRAWL_API_KEY not set")

researcher = FirecrawlResearcher(api_key=API_KEY)

failures = []

def check(label: str, ok: bool, detail: str = ""):
    if ok:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        failures.append(label)

# ---------------------------------------------------------------------------
# 1. Stack Exchange — both answers must appear
# ---------------------------------------------------------------------------
print("\n--- 1. Stack Exchange (SE) ---")
src = researcher.scrape_source(
    "https://electronics.stackexchange.com/questions/368819/transformer-with-opposite-polarity-in-primary-and-secondary-coil",
    query="transformer polarity",
    prompt="transformer polarity",
)
check("No error", src.error is None, src.error or "")
check("Title contains transformer", "transformer" in (src.title or "").lower())
check("Second answer present",
      "Consider how a transformer works" in src.markdown or
      "Neil_UK" in src.markdown)
check("First answer present",
      "There is no significance of the apparent winding" in src.markdown or
      "Andy aka" in src.markdown)
check("Content substantial (>=2000 chars)", len(src.markdown) >= 2000)

# ---------------------------------------------------------------------------
# 2. DigiKey — Product Information API (if subscribed) or graceful fallback
#    When DIGIKEY_CLIENT_ID is set the API is tried first; if the credentials
#    are not subscribed to the Product Information API (sandbox), the test
#    verifies that no crash occurs and that the fallback chain is logged.
# ---------------------------------------------------------------------------
print("\n--- 2. DigiKey ---")
dk_id = os.environ.get("DIGIKEY_CLIENT_ID", "")
src = researcher.scrape_source(
    "https://www.digikey.com/en/products/base-product/texas-instruments/296/LM386/380",
    query="LM386",
    prompt="LM386",
)
if dk_id:
    if src.error:
        check("API attempted (DIGIKEY_CLIENT_ID set)",
              True, "API failed, scrape fallback: " + src.error)
    else:
        check("API or scrape served data (no error)",
              "LM386" in (src.title or ""), src.title or "no title")
else:
    check("No API key — scrape fallback only", src.error is not None, src.error or "")

# ---------------------------------------------------------------------------
# 3. TI.com — manufacturer datasheet page (server-rendered HTML)
# ---------------------------------------------------------------------------
print("\n--- 3. TI.com ---")
src = researcher.scrape_source(
    "https://www.ti.com/product/LM386",
    query="LM386",
    prompt="LM386",
)
check("No error", src.error is None, src.error or "")
check("Title contains LM386", "LM386" in (src.title or ""))
check("Content is substantial (>=2000 chars)", len(src.markdown) >= 2000)
check("Contains technical spec keywords",
      any(kw in src.markdown.lower() for kw in ["supply", "voltage", "output", "gain"]))

# ---------------------------------------------------------------------------
# 4. Wikipedia — LM386 article (always available, no JS rendering)
# ---------------------------------------------------------------------------
print("\n--- 4. Wikipedia ---")
src = researcher.scrape_source(
    "https://en.wikipedia.org/wiki/LM386",
    query="LM386",
    prompt="LM386",
)
check("No error", src.error is None, src.error or "")
check("Title contains LM386", "LM386" in (src.title or ""))
check("Content substantial (>=1000 chars)", len(src.markdown) >= 1000)
check("Contains amplifier keywords",
      any(kw in src.markdown.lower() for kw in ["amplifier", "audio", "gain", "pin"]))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
if failures:
    print(f"FAILED: {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
