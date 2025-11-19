#!/usr/bin/env python3
# Usage examples:
#   # Auto-mirror to cleaned_data/app/eulogies/rows_norm.{parquet,csv}
#   python scripts/normalize_app.py --in data/app/eulogies/rows.csv --drop-notes --flatten-transcript
#
#   # Or specify an explicit base filename (both CSV+Parquet will be written)
#   python scripts/normalize_app.py --in data/app/eulogies/rows.csv --out cleaned_data/app/eulogies/eulogies_norm

import argparse, re, html
from pathlib import Path
import pandas as pd

WS = re.compile(r"\s+")
PUN = re.compile(r"(^[^\w]+|[^\w]+$)")
DATE_TAIL = re.compile(r"online by gerhard peters.*?$", re.IGNORECASE)
NODE_TAIL = re.compile(r"https?://www\.presidency\.ucsb\.edu/node/\d+\s*$", re.IGNORECASE)
NOTE_PARA = re.compile(r"^\s*note:\s*", re.IGNORECASE)

def squash_quotes_and_dashes(s: str) -> str:
    trans = {
        ord("“"): '"', ord("”"): '"', ord("„"): '"', ord("‟"): '"',
        ord("’"): "'", ord("‘"): "'", ord("‚"): "'", ord("‛"): "'",
        ord("—"): "-", ord("–"): "-", ord("−"): "-",
        ord("…"): "...",
        ord("\u00a0"): " ", ord("\u2009"): " ", ord("\u200a"): " ",
        ord("\u202f"): " ", ord("\u200b"): " ",
    }
    return s.translate(trans)

def strip_app_boilerplate(s: str) -> str:
    s = DATE_TAIL.sub("", s)
    s = NODE_TAIL.sub("", s)
    return s.strip()

def maybe_drop_notes(text: str, drop_notes: bool) -> str:
    if not drop_notes:
        return text
    paras = [p.strip() for p in re.split(r"\n{2,}", text)]
    kept = [p for p in paras if not NOTE_PARA.match(p)]
    return "\n\n".join(kept).strip() if kept else ""

def normalize_for_matching(raw: str, drop_notes=False) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    s = html.unescape(raw)
    s = squash_quotes_and_dashes(s)
    s = strip_app_boilerplate(s)
    s = maybe_drop_notes(s, drop_notes)
    s = WS.sub(" ", s)
    s = s.lower().strip()
    tokens = []
    for tok in s.split():
        tok = PUN.sub("", tok)
        if tok:
            tokens.append(tok)
    return " ".join(tokens)

def infer_out_base(in_path: Path) -> Path:
    """
    Mirror data/... to cleaned_data/... and append _norm to stem.
    If input isn't under data/, still place under cleaned_data/ preserving tail.
    """
    p = in_path.resolve()
    parts = list(p.parts)
    try:
        idx = parts.index("data")
        tail = Path(*parts[idx+1:])  # e.g., app/eulogies/rows.csv
    except ValueError:
        # not under data/, just use filename & parent tail
        tail = Path(p.name)
    # flip top-level dir to cleaned_data
    out_dir = Path("cleaned_data") / tail.parent  # e.g., cleaned_data/app/eulogies
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{tail.stem}_norm"          # base filename without extension(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV (scraper output).")
    ap.add_argument("--out", dest="out_base", default=None,
                    help="Output base path WITHOUT extension. Writes both .parquet and .csv. "
                         "If omitted, mirrors under cleaned_data/... with *_norm suffix.")
    ap.add_argument("--flatten-transcript", action="store_true",
                    help="Also include transcript_flat (single-line).")
    ap.add_argument("--drop-notes", action="store_true",
                    help="Remove paragraphs starting with 'Note:' before normalization.")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)

    # Construct a stable doc_id
    if "node_id" in df and df["node_id"].notna().any():
        df["doc_id"] = df["node_id"].fillna("").astype(str).str.strip()
    else:
        df["doc_id"] = pd.util.hash_pandas_object(df["url"].fillna("")).astype("int64").astype(str)

    df["transcript_raw"] = df["transcript"]

    if args.flatten_transcript:
        df["transcript_flat"] = (
            df["transcript_raw"].fillna("").str.replace(r"\s*\n\s*", " ", regex=True).str.strip()
        )

    df["transcript_norm"] = df["transcript_raw"].fillna("").apply(
        lambda t: normalize_for_matching(t, drop_notes=args.drop_notes)
    )
    df["token_count_norm"] = df["transcript_norm"].str.split().str.len()

    keep = ["doc_id","title","president","date_iso","date_text","location","url",
            "word_count","has_video_flag","transcript_raw","transcript_norm","token_count_norm"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    # Figure out output base
    if args.out_base:
        out_base = Path(args.out_base)
        out_base.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_base = infer_out_base(in_path)

    csv_path = out_base.with_suffix(".csv")
    pq_path  = out_base.with_suffix(".parquet")

    # Write CSV (always)
    df.to_csv(csv_path, index=False)

    # Write Parquet (best-effort)
    try:
        df.to_parquet(pq_path, index=False)
        print(f"Wrote {len(df)} docs → {csv_path}, {pq_path}")
    except Exception as e:
        print(f"Wrote {len(df)} docs → {csv_path} (parquet skipped: {e})")

if __name__ == "__main__":
    main()
