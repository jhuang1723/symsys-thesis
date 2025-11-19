#!/usr/bin/env python3
# verse_match_pipeline_app.py
# End-to-end pipeline for APP data (cleaned_data/*).
#
# Steps:
#   1) Index Bible
#      python scripts/verse_match_pipeline_app.py index-bible \
#         --bible_in cleaned_data/bible_kjv_clean.csv --out_dir results/ --translation KJV
#
#   2) Index speeches + make windows
#      python scripts/verse_match_pipeline_app.py index-speeches \
#         --speeches_in cleaned_data/app/eulogies/rows_norm.parquet \
#         --out_dir results/app/eulogies --window_len 30 --stride 5
#
#   3) Generate TF-IDF candidates (TOP-K = 10 by default)
#      python scripts/verse_match_pipeline_app.py gen-candidates \
#         --bible_index results/verses_index.parquet \
#         --windows results/app/eulogies/windows.parquet \
#         --out_dir results/app/eulogies --ngram_min 3 --ngram_max 5 --topk 10
#
#   4) Merge windows → spans, add alignment features, filter, export
#      python scripts/verse_match_pipeline_app.py merge-spans \
#         --out_dir results/app/eulogies --ngram_min 3 --ngram_max 5 \
#         --bible_index results/verses_index.parquet \
#         --min_cov 0.55 --min_lcs 0.45 --max_gap 8 --top_per_doc 10
#
# Outputs (per --out_dir):
#   verses_index.parquet, speeches_index.parquet, windows.parquet,
#   candidates.parquet, matches.parquet, matches_preview.csv

import argparse
import hashlib
import html
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Optional deps
try:
    import pyarrow  # noqa: F401
    _HAVE_PARQUET = True
except Exception:
    _HAVE_PARQUET = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


# -------------------------
# Text normalization / tokenization
# -------------------------

WORD_RE = re.compile(r"(?u)\b\w[\w'-]*\b")  # keep inner ' and -

def normalize_text(s: str) -> str:
    """
    Used for Bible verses. (Speeches already provide transcript_norm.)
    Policy:
      - Unicode NFKC
      - HTML unescape
      - lowercase
      - tokenize with WORD_RE, join by space
    """
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    tokens = WORD_RE.findall(s)
    return " ".join(tokens)


# -------------------------
# IDs
# -------------------------

def stable_hash(*parts: str, n=16) -> str:
    h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()
    return h[:n]


# -------------------------
# IO helpers
# -------------------------

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def write_table(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet" and _HAVE_PARQUET:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".parquet" and not _HAVE_PARQUET:
        csv_path = path.with_suffix(".csv")
        print(f"[warn] pyarrow not found; writing CSV instead -> {csv_path}")
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(path, index=False)

def _read_infer(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


# -------------------------
# 1) Index Bible (schema + normalization)
# -------------------------

def cmd_index_bible(args):
    inp = Path(args.bible_in)
    out_dir = Path(args.out_dir)
    df = read_table(inp)

    # Expect at least: book, chapter, verse, ref, text
    cols = {c.lower(): c for c in df.columns}
    for need in ["book", "chapter", "verse", "ref", "text"]:
        if need not in cols:
            raise ValueError(f"Missing required column '{need}' in {inp}")

    book = df[cols["book"]].astype(str)
    chapter = df[cols["chapter"]].astype(int)
    verse = df[cols["verse"]].astype(int)
    ref = df[cols["ref"]].astype(str)
    text_raw = df[cols["text"]].astype(str)

    text_norm = text_raw.map(normalize_text)
    translation = args.translation
    verse_id = [stable_hash(translation, r) for r in ref]

    out = pd.DataFrame({
        "translation": translation,
        "book": book,
        "chapter": chapter,
        "verse": verse,
        "ref": ref,
        "verse_id": verse_id,
        "text_raw": text_raw,
        "text_norm": text_norm,
        "token_count": text_norm.str.split().map(len)
    })

    write_table(out, out_dir / "verses_index.parquet")
    print(f"[ok] verses_index -> {out_dir/'verses_index.parquet'}  ({len(out)} rows)")


# -------------------------
# 2) Index Speeches (APP cleaned schema) + Windowing (L,S)
# -------------------------

def make_windows(tokens: List[str], L: int, S: int) -> List[Tuple[int,int]]:
    if L <= 0 or S <= 0:
        return []
    windows = []
    i = 0
    n = len(tokens)
    if n == 0:
        return []
    while i < n:
        j = min(i + L, n)
        windows.append((i, j))
        if j == n:
            break
        i += S
    return windows

def cmd_index_speeches(args):
    inp = Path(args.speeches_in)
    out_dir = Path(args.out_dir)
    df = read_table(inp)

    # Expect APP-cleaned columns:
    #   doc_id, title, president, date_iso, transcript_norm
    cols = {c.lower(): c for c in df.columns}
    required = ["doc_id", "title", "president", "date_iso", "transcript_norm"]
    for need in required:
        if need not in cols:
            raise ValueError(f"Missing required column '{need}' in {inp}")

    doc_id = df[cols["doc_id"]].astype(str)
    title = df[cols["title"]].astype(str)
    president = df[cols["president"]].astype(str)
    date = pd.to_datetime(df[cols["date_iso"]], errors="coerce")
    text_norm = df[cols["transcript_norm"]].astype(str)

    # Optional columns
    url_col = cols.get("url")
    url = df[url_col].astype(str) if url_col else None
    raw_col = cols.get("transcript_raw")
    text_raw = df[raw_col].astype(str) if raw_col else None

    meta_cols = {
        "doc_id": doc_id,
        "title": title,
        "president": president,
        "date": date,
        "text_norm": text_norm,
        "token_count": text_norm.str.split().map(len)
    }
    if url is not None:
        meta_cols["url"] = url

    meta = pd.DataFrame(meta_cols)
    write_table(meta, out_dir / "speeches_index.parquet")
    print(f"[ok] speeches_index -> {out_dir/'speeches_index.parquet'}  ({len(meta)} docs)")

    # Windowing
    L, S = args.window_len, args.stride
    records = []
    for row in meta.itertuples(index=False):
        toks = row.text_norm.split()
        spans = make_windows(toks, L, S)
        for (a, b) in spans:
            snippet_norm = " ".join(toks[a:b])
            rec = {
                "window_id": f"{row.doc_id}_{a}_{b}",
                "doc_id": row.doc_id,
                "start_token": a,
                "end_token": b,
                "window_len": b - a,
                "snippet_norm": snippet_norm,
            }
            # add a small raw snippet for the very first window (optional)
            if text_raw is not None and a == 0:
                try:
                    raw_val = text_raw[meta.index[meta["doc_id"] == row.doc_id][0]]
                except Exception:
                    raw_val = None
                if isinstance(raw_val, str):
                    rec["snippet_raw"] = raw_val[:300].replace("\n", " ")
            records.append(rec)

    windows_df = pd.DataFrame.from_records(records)
    write_table(windows_df, out_dir / "windows.parquet")
    print(f"[ok] windows -> {out_dir/'windows.parquet'}  ({len(windows_df)} windows)")


# -------------------------
# 3) Candidate generation (TF-IDF 3–5 grams, TOP-K=10 default)
# -------------------------

def cmd_gen_candidates(args):
    if not _HAVE_SK:
        raise RuntimeError("scikit-learn not available. Install scikit-learn to run gen-candidates.")

    verses = _read_infer(Path(args.bible_index))
    windows = _read_infer(Path(args.windows))

    nmin, nmax = args.ngram_min, args.ngram_max

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(nmin, nmax),
        lowercase=False,                       # already normalized
        token_pattern=r"(?u)\b\w[\w'-]*\b",
        norm="l2",
        sublinear_tf=True,
        smooth_idf=True,
        min_df=1
    )

    corpus = pd.concat([
        verses["text_norm"].astype(str),
        windows["snippet_norm"].astype(str)
    ], ignore_index=True)

    print("[info] fitting TF-IDF vocabulary on verses + windows ...")
    vec.fit(corpus.values)

    print("[info] transforming verses ...")
    B = vec.transform(verses["text_norm"].astype(str).values)      # verses x V
    print("[info] transforming windows ...")
    W = vec.transform(windows["snippet_norm"].astype(str).values)  # windows x V

    topk = args.topk
    chunk = args.batch_size

    win_ids = windows["window_id"].tolist()
    verse_ids = verses["verse_id"].tolist()
    verse_refs = verses["ref"].tolist()

    out_rows = []
    print("[info] scoring (in batches) ...")
    for start in range(0, W.shape[0], chunk):
        end = min(start + chunk, W.shape[0])
        W_block = W[start:end]                        # (m x V)
        scores = W_block @ B.T                        # sparse (m x n)
        for i in range(scores.shape[0]):
            row = scores.getrow(i)
            if row.nnz == 0:
                continue
            data = row.data
            indices = row.indices
            if len(data) <= topk:
                top_idx = np.argsort(-data)
            else:
                part = np.argpartition(-data, topk-1)[:topk]
                top_idx = part[np.argsort(-data[part])]
            for j_pos, j in enumerate(top_idx, start=1):
                out_rows.append({
                    "window_id": win_ids[start + i],
                    "verse_id": verse_ids[indices[j]],
                    "ref": verse_refs[indices[j]],
                    "score": float(data[j]),
                    "rank": j_pos
                })
        if (start // chunk) % 10 == 0:
            print(f"  processed windows {start}..{end} / {W.shape[0]}")

    cand = pd.DataFrame(out_rows)
    cand.sort_values(["window_id","rank"], inplace=True)
    write_table(cand, Path(args.out_dir) / "candidates.parquet")
    print(f"[ok] candidates -> {Path(args.out_dir)/'candidates.parquet'}  ({len(cand)} rows)")


# -------------------------
# 4) Merge overlapping windows → spans, features, filter
# -------------------------

def _ngrams(tokens: List[str], nmin=3, nmax=5):
    for n in range(nmin, nmax+1):
        for i in range(0, max(0, len(tokens)-n+1)):
            yield " ".join(tokens[i:i+n])

def _lcs_len(a_tokens: List[str], b_tokens: List[str]) -> int:
    # token-level LCS (O(n*m))
    n, m = len(a_tokens), len(b_tokens)
    dp = [0]*(m+1)
    for i in range(1, n+1):
        prev = 0
        ai = a_tokens[i-1]
        for j in range(1, m+1):
            tmp = dp[j]
            if ai == b_tokens[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = dp[j] if dp[j] >= dp[j-1] else dp[j-1]
            prev = tmp
    return dp[m]

def _join_all_for_merge(cand_path: Path, wins_path: Path, vers_path: Path, docs_path: Path) -> pd.DataFrame:
    cand = _read_infer(cand_path)[["window_id","verse_id","score"]]
    wins = _read_infer(wins_path)[["window_id","doc_id","start_token","end_token","snippet_norm"]]
    vers = _read_infer(vers_path)[["verse_id","ref","text_raw","text_norm"]]
    docs = _read_infer(docs_path)[["doc_id","title","president","date","text_norm"]]

    df = (cand
          .merge(wins, on="window_id", how="inner")
          .merge(vers.rename(columns={"text_raw":"verse_text","text_norm":"verse_norm"}), on="verse_id", how="inner")
          .merge(docs.rename(columns={"text_norm":"doc_norm"}), on="doc_id", how="inner"))
    return df

def _merge_overlaps(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    merged = []
    key_cols = ["doc_id","verse_id"]
    df = df.sort_values(key_cols + ["start_token"])
    for (doc_id, verse_id), g in df.groupby(key_cols, sort=False):
        cur_start = cur_end = None
        best = 0.0
        for r in g.itertuples(index=False):
            s, e, sc = r.start_token, r.end_token, r.score
            if cur_start is None:
                cur_start, cur_end, best = s, e, sc
            else:
                if s <= cur_end + max_gap:
                    cur_end = max(cur_end, e)
                    if sc > best:
                        best = sc
                else:
                    merged.append((doc_id, verse_id, cur_start, cur_end, best))
                    cur_start, cur_end, best = s, e, sc
        if cur_start is not None:
            merged.append((doc_id, verse_id, cur_start, cur_end, best))
    return pd.DataFrame(merged, columns=["doc_id","verse_id","start_token","end_token","score_max"])

def _add_text_and_features(spans: pd.DataFrame, docs: pd.DataFrame, vers: pd.DataFrame, nmin: int, nmax: int) -> pd.DataFrame:
    docs_idx = docs.set_index("doc_id")
    vers_idx = vers.set_index("verse_id")

    rows = []
    for r in spans.itertuples(index=False):
        dn = docs_idx.at[r.doc_id, "doc_norm"]
        doc_tokens = dn.split()
        snip_tokens = doc_tokens[r.start_token:r.end_token]
        snippet_norm = " ".join(snip_tokens)

        vref     = vers_idx.at[r.verse_id, "ref"]
        vtext    = vers_idx.at[r.verse_id, "verse_text"]
        vnorm    = vers_idx.at[r.verse_id, "verse_norm"]
        v_tokens = vnorm.split()

        sn_set = set(_ngrams(snip_tokens, nmin, nmax))
        v_set  = set(_ngrams(v_tokens, nmin, nmax))
        inter  = sn_set & v_set
        cov    = len(inter) / max(1, len(v_set))

        lcs    = _lcs_len(snip_tokens, v_tokens)
        lcsr   = lcs / max(1, len(v_tokens))

        rows.append({
            "doc_id": r.doc_id,
            "verse_id": r.verse_id,
            "start_token": r.start_token,
            "end_token": r.end_token,
            "score_max": r.score_max,
            "snippet_norm": snippet_norm,
            "ref": vref,
            "verse_text": vtext,
            "cov_ngram": cov,
            "lcs_ratio": lcsr,
        })

    out = pd.DataFrame(rows)
    out = out.merge(docs_idx[["title","president","date","doc_norm"]].reset_index(), on="doc_id", how="left")
    return out

def cmd_merge_spans(args):
    out_dir = Path(args.out_dir)

    cand_path  = Path(args.candidates) if args.candidates else (out_dir / "candidates.parquet")
    wins_path  = Path(args.windows) if args.windows else (out_dir / "windows.parquet")
    vers_path  = Path(args.bible_index) if args.bible_index else (out_dir / "verses_index.parquet")
    docs_path  = Path(args.speeches_index) if args.speeches_index else (out_dir / "speeches_index.parquet")

    for pth in [cand_path, wins_path, vers_path, docs_path]:
        if not pth.exists():
            raise FileNotFoundError(f"Missing input at {pth}")

    df = _join_all_for_merge(cand_path, wins_path, vers_path, docs_path)

    spans = _merge_overlaps(df[["doc_id","verse_id","start_token","end_token","score"]], max_gap=args.max_gap)

    docs = _read_infer(docs_path)[["doc_id","title","president","date","text_norm"]].rename(columns={"text_norm":"doc_norm"})
    vers = _read_infer(vers_path)[["verse_id","ref","text_raw","text_norm"]].rename(columns={"text_raw":"verse_text","text_norm":"verse_norm"})
    merged = _add_text_and_features(spans, docs, vers, args.ngram_min, args.ngram_max)

    keep = merged[(merged["cov_ngram"] >= args.min_cov) | (merged["lcs_ratio"] >= args.min_lcs)]

    keep = keep.sort_values(["doc_id","verse_id","score_max"], ascending=[True, True, False]) \
               .drop_duplicates(subset=["doc_id","verse_id"], keep="first")

    keep["rank_in_doc"] = keep.groupby("doc_id")["score_max"].rank(method="first", ascending=False)
    keep_top = keep[keep["rank_in_doc"] <= args.top_per_doc] \
        .sort_values(["doc_id","score_max"], ascending=[True, False])

    write_table(keep, out_dir / "matches.parquet")
    keep_top[[
        "doc_id","title","president","date","ref","score_max",
        "cov_ngram","lcs_ratio","start_token","end_token","snippet_norm","verse_text"
    ]].to_csv(out_dir / "matches_preview.csv", index=False)

    print(f"[ok] matches -> {out_dir/'matches.parquet'}  ({len(keep)} rows)")
    print(f"[ok] preview -> {out_dir/'matches_preview.csv'}  ({len(keep_top)} rows)")


# -------------------------
# CLI
# -------------------------

def main():
    p = argparse.ArgumentParser(prog="verse_match_pipeline_app", description="Bible-verse ↔ APP speech pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("index-bible", help="Normalize and index Bible verses")
    p1.add_argument("--bible_in", required=True, help="CSV or Parquet with columns: book,chapter,verse,ref,text")
    p1.add_argument("--translation", default="KJV", help="Translation tag (default: KJV)")
    p1.add_argument("--out_dir", required=True)
    p1.set_defaults(func=cmd_index_bible)

    p2 = sub.add_parser("index-speeches", help="Index APP speeches (use transcript_norm) and make windows")
    p2.add_argument("--speeches_in", required=True, help="cleaned_data/app/.../rows_norm.{csv|parquet}")
    p2.add_argument("--out_dir", required=True)
    p2.add_argument("--window_len", type=int, default=30)
    p2.add_argument("--stride", type=int, default=5)
    p2.set_defaults(func=cmd_index_speeches)

    p3 = sub.add_parser("gen-candidates", help="Generate top-k verse candidates per window via TF-IDF")
    p3.add_argument("--bible_index", required=True, help="verses_index.parquet (from index-bible)")
    p3.add_argument("--windows", required=True, help="windows.parquet (from index-speeches)")
    p3.add_argument("--out_dir", required=True)
    p3.add_argument("--ngram_min", type=int, default=3)
    p3.add_argument("--ngram_max", type=int, default=5)
    p3.add_argument("--topk", type=int, default=10)          # <-- K=10 default
    p3.add_argument("--batch_size", type=int, default=1000, help="windows per batch")
    p3.set_defaults(func=cmd_gen_candidates)

    p4 = sub.add_parser("merge-spans", help="Merge overlapping windows per verse, add features, filter, export")
    p4.add_argument("--out_dir", required=True, help="Directory containing candidates/windows/verses_index/speeches_index")
    p4.add_argument("--candidates", help="Optional explicit path to candidates file; defaults to OUT_DIR/candidates.parquet")
    p4.add_argument("--windows", help="Optional explicit path to windows file; defaults to OUT_DIR/windows.parquet")
    p4.add_argument("--bible_index", help="Optional path to verses_index; defaults to OUT_DIR/verses_index.parquet")
    p4.add_argument("--speeches_index", help="Optional path to speeches_index; defaults to OUT_DIR/speeches_index.parquet")
    p4.add_argument("--ngram_min", type=int, default=3)
    p4.add_argument("--ngram_max", type=int, default=5)
    p4.add_argument("--min_cov", type=float, default=0.55, help="min verse n-gram coverage to keep")
    p4.add_argument("--min_lcs", type=float, default=0.45, help="min token-level LCS ratio to keep")
    p4.add_argument("--max_gap", type=int, default=8, help="max token gap when merging adjacent windows")
    p4.add_argument("--top_per_doc", type=int, default=10, help="how many merged hits to keep per doc for preview")
    p4.set_defaults(func=cmd_merge_spans)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
