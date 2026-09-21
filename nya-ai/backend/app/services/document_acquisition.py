"""Document Acquisition Service.

Handles fetching legal judgments from Indian Kanoon or other court URLs with:
- Polite User-Agent headers
- Rate limiting and backoff
- SHA-256 document hashing
- Local disk caching to prevent duplicate downloads
- Timeout and error handling
"""
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from bs4 import BeautifulSoup

from .text_cleaner import clean_legal_text

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "document_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 NyaAI-LegalResearch/1.0"


def url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]


def acquire_document_from_url(url: str, max_retries: int = 2, timeout: float = 12.0) -> Dict[str, Any]:
    """Retrieves document from URL, extracts content, and returns structured result.
    
    Returns:
    {
        "status": "acquired" | "cached" | "failed",
        "url": url,
        "raw_text": str,
        "clean_text": str,
        "document_hash": str,
        "title": str,
        "error": Optional[str]
    }
    """
    if not url or not url.startswith("http"):
        return {
            "status": "failed",
            "url": url,
            "raw_text": "",
            "clean_text": "",
            "document_hash": "",
            "title": "",
            "error": "Invalid URL provided.",
        }

    h = url_hash(url)
    cache_file = CACHE_DIR / f"{h}.txt"

    # Check local disk cache first
    if cache_file.exists():
        try:
            cached_text = cache_file.read_text(encoding="utf-8")
            return {
                "status": "cached",
                "url": url,
                "raw_text": cached_text,
                "clean_text": clean_legal_text(cached_text),
                "document_hash": hashlib.sha256(cached_text.encode("utf-8")).hexdigest(),
                "title": _extract_title_from_text(cached_text),
                "error": None,
            }
        except Exception as e:
            logger.warning(f"Cache read error for {url}: {e}")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
                response = client.get(url)
                if response.status_code == 200:
                    html_content = response.text
                    extracted_text, title = _parse_indian_kanoon_or_html(html_content)
                    
                    if not extracted_text or len(extracted_text.strip()) < 40:
                        raise ValueError("Extracted text was too short or empty.")

                    # Write to cache
                    try:
                        cache_file.write_text(extracted_text, encoding="utf-8")
                    except Exception as cache_err:
                        logger.warning(f"Failed to cache document: {cache_err}")

                    doc_hash = hashlib.sha256(extracted_text.encode("utf-8")).hexdigest()
                    return {
                        "status": "acquired",
                        "url": url,
                        "raw_text": extracted_text,
                        "clean_text": clean_legal_text(extracted_text),
                        "document_hash": doc_hash,
                        "title": title or _extract_title_from_text(extracted_text),
                        "error": None,
                    }
                else:
                    last_error = f"HTTP {response.status_code}"
        except Exception as e:
            last_error = str(e)
            time.sleep(1.0 * (attempt + 1))  # exponential-ish backoff

    return {
        "status": "failed",
        "url": url,
        "raw_text": "",
        "clean_text": "",
        "document_hash": "",
        "title": "",
        "error": last_error,
    }


def _parse_indian_kanoon_or_html(html: str) -> tuple[str, str]:
    """Parses HTML judgment page, prioritizing Indian Kanoon content containers."""
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    title_el = soup.find("h2", class_="doc_title") or soup.find("title")
    if title_el:
        title = title_el.get_text().strip()

    # In Indian Kanoon, judgments are typically in div.judgments or div.doc_fragment
    main_div = soup.find("div", class_="judgments") or soup.find("div", id="doc_holder") or soup.find("article")
    if main_div:
        # remove scripts and styles
        for tag in main_div(["script", "style", "nav", "footer"]):
            tag.decompose()
        return main_div.get_text(separator="\n"), title

    # Fallback to general body extraction
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n"), title


def _extract_title_from_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[0][:150]
    return "Acquired Judgment"
