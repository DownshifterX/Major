import hashlib, logging, os, re, time
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
    if not url or not url.startswith("http"):
        return {"status": "failed", "url": url, "raw_text": "", "clean_text": "", "document_hash": "", "title": "", "error": "Invalid URL"}

    uhash = url_hash(url)
    cache_txt = CACHE_DIR / f"{uhash}.txt"
    cache_meta = CACHE_DIR / f"{uhash}_meta.json"

    if cache_txt.exists() and cache_meta.exists():
        try:
            import json
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            clean_txt = cache_txt.read_text(encoding="utf-8")
            return {
                "status": "cached", "url": url, "raw_text": clean_txt, "clean_text": clean_txt,
                "document_hash": uhash, "title": meta.get("title", "Cached Case"),
                "metadata": meta.get("metadata", {}), "kanoon_structure": meta.get("kanoon_structure", {}), "error": None
            }
        except Exception as e:
            logger.warning(f"Failed to read cache for {url}: {e}")

    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    resp = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
                r = client.get(url)
                if r.status_code == 200:
                    resp = r; break
                elif r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
        except Exception as exc:
            if attempt == max_retries:
                return {"status": "failed", "url": url, "raw_text": "", "clean_text": "", "document_hash": uhash, "title": "", "error": str(exc)}
            time.sleep(1.0)

    if not resp or resp.status_code != 200:
        code = resp.status_code if resp else "timeout"
        return {"status": "failed", "url": url, "raw_text": "", "clean_text": "", "document_hash": uhash, "title": "", "error": f"HTTP {code}"}

    raw_text, title, kanoon_struct = _parse_indian_kanoon_or_html(resp.text)
    clean_txt = clean_legal_text(raw_text)

    try:
        import json
        cache_txt.write_text(clean_txt, encoding="utf-8")
        cache_meta.write_text(json.dumps({"title": title, "url": url, "kanoon_structure": kanoon_struct}, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

    return {
        "status": "acquired", "url": url, "raw_text": raw_text, "clean_text": clean_txt,
        "document_hash": uhash, "title": title, "metadata": {}, "kanoon_structure": kanoon_struct, "error": None
    }

def _parse_indian_kanoon_or_html(html: str) -> tuple[str, str, Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    title_tag = soup.find("title") or soup.find("h2", class_="doc_title") or soup.find("h1")
    if title_tag:
        title = re.sub(r"\s+", " ", title_tag.get_text()).replace(" - Indian Kanoon", "").strip()

    k_struct: Dict[str, list] = {"facts": [], "issues": [], "petitioner_arguments": [], "respondent_arguments": [], "decision": []}
    for p in soup.find_all("p"):
        txt = p.get_text().strip()
        cid = p.get("id", "")
        if "fact" in cid: k_struct["facts"].append(txt)
        elif "issue" in cid: k_struct["issues"].append(txt)
        elif "pet" in cid: k_struct["petitioner_arguments"].append(txt)
        elif "resp" in cid: k_struct["respondent_arguments"].append(txt)
        elif "dec" in cid or "order" in cid: k_struct["decision"].append(txt)

    content_div = soup.find("div", class_="judgments") or soup.find("div", class_="doc_content") or soup.find("body") or soup
    for rem in content_div(["script", "style", "nav", "footer", "form", "button"]):
        rem.decompose()

    text = content_div.get_text(separator="\n\n")
    if not title:
        title = _extract_title_from_text(text)
    return text, title, k_struct

def _extract_title_from_text(text: str) -> str:
    for line in text.splitlines()[:20]:
        s = line.strip()
        if re.search(r"\b(?:vs\.?|versus)\b", s, re.IGNORECASE) and 10 < len(s) < 120:
            return s
    return "Acquired Indian Kanoon Judgment"
