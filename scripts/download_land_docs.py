"""
download_land_docs.py
=====================
Downloads all PDF/HTML documents from the `Doc_url` column of
land_disputes.xlsx into:
    nya-ai/downloaded_cases/

Features
--------
- Skips already-downloaded files (resume-safe)
- Parallel downloads with a thread pool (configurable)
- Sanitises filenames from titles
- Shows a live progress bar (tqdm)
- Saves a manifest CSV: downloaded_cases/manifest.csv
- Logs failures to: downloaded_cases/failed.csv

Usage
-----
    python download_land_docs.py                  # default 8 threads
    python download_land_docs.py --workers 16     # more threads
    python download_land_docs.py --limit 500      # download first N rows only
    python download_land_docs.py --dry-run        # just print URLs, no download
"""

import os
import re
import csv
import time
import argparse
import hashlib
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
EXCEL_PATH   = SCRIPT_DIR.parent / "dataset - hugging face" / "land_disputes.xlsx"
OUTPUT_DIR   = SCRIPT_DIR / "downloaded_cases"
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"
FAILED_CSV   = OUTPUT_DIR / "failed.csv"

TIMEOUT      = 30          # seconds per request
MAX_RETRIES  = 3
RETRY_DELAY  = 2           # seconds between retries
CHUNK_SIZE   = 1024 * 64   # 64 KB streaming chunks

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html,*/*",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "download.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def slugify(text: str, max_len: int = 80) -> str:
    """Convert a title to a safe filename slug."""
    text = str(text)
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s]+', '_', text.strip())
    return text[:max_len]


def url_to_filename(title: str, url: str, idx: int) -> str:
    """
    Build a deterministic filename:
      <idx>_<slug>.<ext>
    Extension is guessed from the URL; defaults to .pdf
    """
    slug = slugify(title, 60) if title and str(title) not in ("nan", "") else f"case_{idx}"
    url_lower = url.lower().split("?")[0]
    if url_lower.endswith(".html") or url_lower.endswith(".htm"):
        ext = ".html"
    else:
        ext = ".pdf"   # IndianKanoon and most legal portals serve PDFs
    uid = hashlib.md5(url.encode()).hexdigest()[:6]
    return f"{idx:05d}_{slug}_{uid}{ext}"


def download_one(row_idx: int, title: str, url: str, out_dir: Path, dry_run: bool):
    """Download a single document. Returns a dict describing the result."""
    filename = url_to_filename(title, url, row_idx)
    dest = out_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        return {
            "idx": row_idx, "title": title, "url": url,
            "file": filename, "status": "skipped", "bytes": dest.stat().st_size,
        }

    if dry_run:
        return {
            "idx": row_idx, "title": title, "url": url,
            "file": filename, "status": "dry-run", "bytes": 0,
        }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()

            # Adjust extension based on server content-type
            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type and not filename.endswith(".html"):
                dest = dest.with_suffix(".html")
                filename = dest.name

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)

            size = dest.stat().st_size
            return {
                "idx": row_idx, "title": title, "url": url,
                "file": filename, "status": "ok", "bytes": size,
            }

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return {
                    "idx": row_idx, "title": title, "url": url,
                    "file": filename, "status": f"FAILED: {e}", "bytes": 0,
                }


def load_existing_manifest(path: Path) -> set:
    """Return set of URLs already successfully downloaded (for resume)."""
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["url"] for row in csv.DictReader(f) if row.get("status") == "ok"}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download land-dispute case documents.")
    parser.add_argument("--workers",  type=int, default=8,    help="Parallel download threads (default: 8)")
    parser.add_argument("--limit",    type=int, default=None, help="Only process first N rows")
    parser.add_argument("--dry-run",  action="store_true",    help="Print URLs without downloading")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    log.info(f"Reading dataset: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH, usecols=["Titles", "Doc_url"])
    df = df.dropna(subset=["Doc_url"])
    df = df[df["Doc_url"].str.startswith("http", na=False)]

    if args.limit:
        df = df.head(args.limit)
        log.info(f"Limited to first {args.limit} rows.")

    total = len(df)
    log.info(f"Total documents in dataset: {total}")

    # ── Skip already downloaded ───────────────────────────────────────────────
    done_urls = load_existing_manifest(MANIFEST_CSV)
    df = df[~df["Doc_url"].isin(done_urls)]
    remaining = len(df)
    log.info(f"Already downloaded: {total - remaining} | Remaining: {remaining}")

    if df.empty:
        log.info("Nothing to download — all files already present!")
        return

    rows = df.reset_index(drop=True).to_dict("records")

    # ── Parallel download ─────────────────────────────────────────────────────
    results = []
    failed  = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one,
                i, str(row.get("Titles", "")), str(row["Doc_url"]),
                OUTPUT_DIR, args.dry_run,
            ): i
            for i, row in enumerate(rows)
        }

        with tqdm(as_completed(futures), total=len(futures),
                  desc="Downloading", unit="file", dynamic_ncols=True) as pbar:
            for fut in pbar:
                res = fut.result()
                results.append(res)
                if "FAILED" in str(res["status"]):
                    failed.append(res)
                pbar.set_postfix(ok=len(results) - len(failed), fail=len(failed))

    # ── Write manifest ────────────────────────────────────────────────────────
    fieldnames = ["idx", "title", "url", "file", "status", "bytes"]
    write_header = not MANIFEST_CSV.exists()
    with open(MANIFEST_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    # ── Write failures ────────────────────────────────────────────────────────
    if failed:
        with open(FAILED_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(failed)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok_count  = sum(1 for r in results if r["status"] == "ok")
    skipped   = sum(1 for r in results if r["status"] == "skipped")
    total_mb  = sum(r["bytes"] for r in results) / (1024 * 1024)

    log.info("=" * 60)
    log.info(f"  Downloaded : {ok_count}")
    log.info(f"  Skipped    : {skipped}")
    log.info(f"  Failed     : {len(failed)}")
    log.info(f"  Total size : {total_mb:.1f} MB")
    log.info(f"  Output dir : {OUTPUT_DIR}")
    log.info(f"  Manifest   : {MANIFEST_CSV}")
    if failed:
        log.info(f"  Failures   : {FAILED_CSV}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
