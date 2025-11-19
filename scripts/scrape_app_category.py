#!/usr/bin/env python3
"""
Scrape a single APP category (e.g., eulogies, farewell-addresses, etc.)
Example:
  python scrape_app_category.py \
    --slug eulogies \
    --items-per-page 60 \
    --out-dir data/app/eulogies \
    --require-transcript
"""
import argparse, time, sys, re
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE = "https://www.presidency.ucsb.edu"
CAT_PREFIX = "/documents/app-categories/spoken-addresses-and-remarks/presidential/"
UA = "APP-Category-Scraper/1.1 (+research; contact: you@example.com)"

def backoff_sleep(i):  # gentle + retry-friendly
    time.sleep(0.5 + 0.5 * i)

def http_get(url, max_retries=5, timeout=30):
    for i in range(max_retries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                backoff_sleep(i)
                continue
            r.raise_for_status()
        except requests.RequestException:
            backoff_sleep(i)
    raise RuntimeError(f"GET failed after retries: {url}")

def norm_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def guess_date_iso(date_text: str):
    if not date_text: return None
    try:
        return pd.to_datetime(date_text).date().isoformat()
    except Exception:
        return None

def iter_category_doc_urls(category_url: str, items_per_page: int, max_pages: int | None):
    """Yield /documents/... urls for the category, following pagination."""
    seen = set()
    page = 0
    while True:
        url = f"{category_url}?items_per_page={items_per_page}&page={page}"
        html = http_get(url).text
        soup = BeautifulSoup(html, "html.parser")

        # Collect document links on the page (defensively: there are many nav links)
        # We take all anchors that look like /documents/<slug> and are in the main content.
        main = soup.find("main") or soup
        for a in main.select('a[href^="/documents/"]'):
            href = a.get("href")
            if not href: 
                continue
            # Ensure it's a document page, not anchors or query-only links
            if href.startswith("/documents/"):
                full = urljoin(BASE, href)
                if full not in seen:
                    seen.add(full)
                    yield full

        # Pagination: stop when there's no "next ›" link text
        pager = soup.find("ul", class_="pagination")
        has_next = False
        if pager and "next" in pager.get_text(" ").lower():
            # Be safe: only advance if a next link exists
            for a in pager.find_all("a"):
                if "next" in a.get_text(" ").lower():
                    has_next = True
                    break

        page += 1
        if (max_pages and page >= max_pages) or not has_next:
            break
        time.sleep(0.4)

def parse_document(doc_url: str) -> dict:
    """Return metadata + transcript (or transcript=None if not present)."""
    html = http_get(doc_url).text
    soup = BeautifulSoup(html, "html.parser")

    # Title
    h1 = soup.find("h1")
    title = norm_whitespace(h1.get_text()) if h1 else None

    # President (usually a link in an h3 above the title)
    president = None
    pres = soup.select_one("h3 a[href^='/president/'], h3 a[href^='/people/'], h3")
    if pres:
        president = norm_whitespace(pres.get_text())

    # Date (prefer <time> tag; fallback to the line after the title)
    date_text = None
    ttag = soup.find("time")
    if ttag and ttag.get_text(strip=True):
        date_text = ttag.get_text(strip=True)
    if not date_text and h1:
        # scan a few sibling strings for Month dd, yyyy
        cur = h1
        for _ in range(8):
            cur = cur.find_next(string=True)
            if not cur: break
            s = str(cur).strip()
            if re.search(r"[A-Za-z]+\s+\d{1,2},\s+\d{4}", s):
                date_text = s
                break
    date_iso = guess_date_iso(date_text)

    # Location (optional)
    location = None
    for tag in soup.find_all(["h3", "h4"]):
        if tag.get_text(strip=True).lower() == "location":
            nxt = tag.find_next()
            if nxt:
                location = norm_whitespace(nxt.get_text(" ", strip=True))
            break

    # Node id (handy stable key), appears in citation block as /node/<id>
    node_id = None
    for a in soup.select("a[href^='/node/']"):
        m = re.search(r"/node/(\d+)", a.get("href", ""))
        if m:
            node_id = m.group(1)
            break

    # Transcript: paragraphs within the main article
    main = soup.find(id="block-system-main") or soup.find("main") or soup
    paras = [norm_whitespace(p.get_text(" ", strip=True)) for p in main.select("p")]
    paras = [p for p in paras if p]
    transcript = "\n\n".join(paras) if paras else None

    # Heuristic: some pages are video-only or otherwise empty; also catch very short bodies
    if transcript:
        wc = len(transcript.split())
    else:
        wc = 0

    # Also useful: flag if the page advertises a video
    has_video_flag = bool(main.find(string=re.compile(r"watch video", re.I)))

    return {
        "title": title,
        "date_text": date_text,
        "date_iso": date_iso,
        "president": president,
        "location": location,
        "url": doc_url,
        "node_id": node_id,
        "word_count": wc or None,
        "has_video_flag": has_video_flag,
        "transcript": transcript,
    }

def build_category_url(slug_or_url: str) -> str:
    if slug_or_url.startswith("http"):
        return slug_or_url
    # accept raw slug like "eulogies" and build full path
    slug = slug_or_url.strip("/ ")
    return urljoin(BASE, CAT_PREFIX + slug)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True,
                    help="Category slug OR full category URL "
                         "(e.g., 'eulogies' or full https://.../presidential/eulogies)")
    ap.add_argument("--items-per-page", type=int, default=60)
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Stop after this many pages (useful for testing).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--require-transcript", action="store_true",
                    help="Skip pages with no transcript text.")
    ap.add_argument("--min-words", type=int, default=30,
                    help="When --require-transcript, also skip if transcript shorter than this.")
    args = ap.parse_args()

    cat_url = build_category_url(args.slug)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen_path = args.out_dir / "seen_urls.txt"

    # resumability
    seen = set()
    if seen_path.exists():
        seen = set(u.strip() for u in seen_path.read_text().splitlines() if u.strip())

    rows = []
    try:
        for i, doc_url in enumerate(iter_category_doc_urls(cat_url, args.items_per_page, args.max_pages), 1):
            if doc_url in seen:
                continue
            try:
                rec = parse_document(doc_url)
                if args.require_transcript:
                    if not rec["transcript"] or (rec["word_count"] or 0) < args.min_words:
                        # Skip video-only or empty pages
                        seen.add(doc_url)
                        if i % 10 == 0: seen_path.write_text("\n".join(sorted(seen)))
                        time.sleep(0.25)
                        continue
                rows.append(rec)
                seen.add(doc_url)
                if i % 10 == 0:
                    seen_path.write_text("\n".join(sorted(seen)))
                time.sleep(0.25)  # politeness
            except Exception as e:
                print(f"[warn] parse failed: {doc_url} ({e})", file=sys.stderr)
                time.sleep(0.8)
    finally:
        # always persist progress
        seen_path.write_text("\n".join(sorted(seen)))

    if not rows:
        print("No rows scraped (maybe --require-transcript filtered them all).")
        return

    df = pd.DataFrame(rows)
    df.sort_values(["date_iso", "president", "title"], inplace=True, na_position="last")

    csv_path = args.out_dir / "rows.csv"
    pq_path = args.out_dir / "rows.parquet"

    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(pq_path, index=False)
        pq_written = True
    except Exception:
        pq_written = False
        print("[info] Parquet not written (install pyarrow or fastparquet).")

    print(f"Wrote {len(df)} rows to:\n- {csv_path}\n- {pq_path if pq_written else '(parquet skipped)'}")

if __name__ == "__main__":
    main()
