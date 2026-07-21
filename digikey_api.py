"""
digikey_api.py
--------------
DigiKey Product Information v4 API client.

Two-legged OAuth2 (client_credentials) flow:
  - Token endpoint: POST /v1/oauth2/token
  - ProductDetails: GET /products/v4/search/{partNumber}/productdetails

Base URL is parameterized — use ``sandbox-api.digikey.com`` for testing
or ``api.digikey.com`` for production.  Access tokens are cached in memory
and automatically refreshed when expired.
"""
import json
import os
import time
from typing import Optional, Dict, Any

import requests


# ── Exceptions ────────────────────────────────────────────────────────────────

class DigiKeyError(Exception):
    """Base for all DigiKey API errors."""
class DigiKeyAuthError(DigiKeyError):
    """OAuth2 token fetch failed (invalid credentials, not subscribed, etc.)."""
class DigiKeyNotFoundError(DigiKeyError):
    """Part number not found."""
class DigiKeyNetworkError(DigiKeyError):
    """Network-level failure (timeout, connection refused, etc.)."""
class DigiKeyApiError(DigiKeyError):
    """API returned an unexpected error status."""


# Sent as User-Agent on all requests — avoids Cloudflare challenges
# that DigiKey's sandbox environment issues to unrecognised clients.
_USER_AGENT = "PCB-Design-Research-Scraper/1.0"


# ── Client ────────────────────────────────────────────────────────────────────

class DigiKeyClient:
    """
    DigiKey Product Information v4 API client.

    Usage::

        client = DigiKeyClient(client_id="...", client_secret="...",
                               base_url="sandbox-api.digikey.com")
        details = client.get_product_details("LM386N-1")
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        base_url: str = "sandbox-api.digikey.com",
    ):
        self.client_id = client_id or os.environ.get("DIGIKEY_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("DIGIKEY_CLIENT_SECRET", "")
        self.base_url = base_url.rstrip("/")
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0  # Unix TS when token expires

    # -- helpers ---------------------------------------------------------------

    def _ensure_token(self) -> str:
        """Return a valid access token, fetching or refreshing if needed."""
        if self._token and time.time() < self._token_expiry - 30:
            return self._token

        token_url = f"https://{self.base_url}/v1/oauth2/token"
        try:
            resp = requests.post(
                token_url,
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials"},
                headers={
                    "User-Agent": _USER_AGENT,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=15,
            )
        except requests.RequestException as e:
            raise DigiKeyNetworkError(
                f"Token endpoint unreachable ({self.base_url}): {e}"
            ) from e

        if resp.status_code != 200:
            raise DigiKeyAuthError(
                f"OAuth2 token fetch failed (HTTP {resp.status_code}): "
                f"{resp.text[:200]}"
            )

        body = resp.json()
        self._token = body["access_token"]
        self._token_expiry = time.time() + body.get("expires_in", 600)
        return self._token

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._ensure_token()}",
            "X-DIGIKEY-Client-Id": self.client_id,
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        }

    # -- ProductDetails --------------------------------------------------------

    def get_product_details(self, part_number: str) -> dict:
        """
        Fetch product details for *part_number* from the Product Information
        v4 API.

        Returns a normalized dict with keys:
          - ``part_number``    — requested part number (normalised)
          - ``description``    — full short description
          - ``manufacturer``   — manufacturer name
          - ``quantity_available`` — int (0 if unknown)
          - ``price_breaks``   — list of ``{qty: int, price: float}`` sorted asc
          - ``datasheet_url``  — URL string or ``None``
          - ``category``       — product category path string

        Raises:
            DigiKeyAuthError   — OAuth2 failure or API subscription issue
            DigiKeyNotFoundError — part not found in DigiKey catalogue
            DigiKeyNetworkError  — network-level failure
            DigiKeyApiError    — other API-level errors
        """
        token = self._ensure_token()
        url = (f"https://{self.base_url}"
               f"/products/v4/search/{_quote_part(part_number)}/productdetails")

        try:
            resp = requests.get(url, headers=self._headers(), timeout=20)
        except requests.RequestException as e:
            raise DigiKeyNetworkError(
                f"ProductDetails request failed: {e}"
            ) from e

        if resp.status_code == 404:
            raise DigiKeyNotFoundError(
                f"Part '{part_number}' not found in DigiKey catalogue"
            )
        if resp.status_code in (401, 403):
            raise DigiKeyAuthError(
                f"DigiKey API returned HTTP {resp.status_code}: "
                f"{resp.json().get('detail', resp.text[:200])}"
            )
        if resp.status_code != 200:
            raise DigiKeyApiError(
                f"ProductDetails returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )

        raw = resp.json()
        return self._normalize(raw, part_number)

    @staticmethod
    def _normalize(raw: dict, requested_pn: str) -> dict:
        """Flatten the verbose API response into a compact dict.

        The V4 ProductDetails response wraps the product under a
        ``Product`` key; this method handles both the wrapped and
        unwrapped shapes.
        """
        p = raw.get("Product") or raw

        price_breaks = []
        # V4 nests price breaks inside each ProductVariations entry
        for var in p.get("ProductVariations", []):
            for pb in var.get("StandardPricing", []):
                qty = pb.get("BreakQuantity", 0)
                price = pb.get("UnitPrice", 0.0)
                if qty and price:
                    price_breaks.append({"qty": qty, "price": float(price)})
        # Fallback: flat priceBreaks list (older / sandbox shape)
        if not price_breaks:
            for pb in p.get("priceBreaks", []):
                qty = pb.get("BreakQuantity", pb.get("breakQuantity", 0))
                price = pb.get("UnitPrice", pb.get("unitPrice", 0.0))
                if qty and price:
                    price_breaks.append({"qty": qty, "price": float(price)})
        price_breaks.sort(key=lambda x: x["qty"])

        desc = p.get("Description") or {}
        manufacturer = p.get("Manufacturer") or {}

        return {
            "part_number": (p.get("DigiKeyProductNumber")
                            or p.get("digiKeyPartNumber")
                            or requested_pn),
            "description": (p.get("productDescription")
                            or p.get("ProductDescription")
                            or desc.get("ProductDescription", "")),
            "manufacturer": (manufacturer.get("Name")
                             or manufacturer.get("name", "")),
            "quantity_available": (p.get("QuantityAvailable", 0)
                                   or p.get("quantityAvailable", 0)),
            "price_breaks": price_breaks,
            "datasheet_url": p.get("DatasheetUrl") or p.get("datasheetUrl"),
            "category": (
                (p.get("category") or p.get("Category") or {}).get("path", "")
            ),
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _quote_part(pn: str) -> str:
    """URL-encode the part number for the path segment."""
    from urllib.parse import quote
    return quote(pn, safe="")

def make_markdown(details: dict) -> str:
    """
    Render a normalized product-details dict into a compact markdown block
    suitable for inserting into a scraped Source's markdown.
    """
    lines = [f"**{details['part_number']}** — {details['description']}"]
    lines.append(f"**Manufacturer:** {details['manufacturer']}")
    lines.append(f"**Available:** {details['quantity_available']:,}")

    if details["price_breaks"]:
        pb_lines = ["**Pricing:**"]
        for pb in details["price_breaks"]:
            pb_lines.append(f"  - {pb['qty']:>6,} @ ${pb['price']:.4f}")
        lines.append("\n".join(pb_lines))

    if details["datasheet_url"]:
        lines.append(f"📄 [Datasheet]({details['datasheet_url']})")

    if details["category"]:
        lines.append(f"**Category:** {details['category']}")

    return "\n\n".join(lines)
