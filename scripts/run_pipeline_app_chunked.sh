#!/usr/bin/env bash
# scripts/run_pipeline_app_chunked.sh
# Chunked pipeline for a single APP category (keeps each chunk separate; no LLM step).
# Usage:
#   bash scripts/run_pipeline_app_chunked.sh <category> [options]
#
# Example:
#   bash scripts/run_pipeline_app_chunked.sh spoken-addresses-and-remarks --chunks 5

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 <category> [options]

Options:
  --chunks N              Number of chunks to split windows into (default: 5)
  --model-path PATH       Path to saved BibleMatchClassifier (default: models/apb_lgbm_sem.pkl)
  --threshold FLOAT       Decision threshold for classifier (default: 0.1)
  --ngram-min N           ngram min for gen-candidates (default: 3)
  --ngram-max N           ngram max for gen-candidates (default: 5)
  --top-per-doc N         how many merged hits to keep per doc for preview (default: 15)
  --tfidf-topk N          TF-IDF candidates to keep per window (default: 25)
  --batch-size N          windows per batch for TF-IDF scoring (default: 1000)
  --out-dir PATH          Override default results dir (default: results/app/<category>)
  --keyword-candidates    Explicitly enable keyword-based candidate generation (default: on)
  --no-keyword-candidates Disable keyword-based candidates
  --keyword-max-total N   Limit keyword candidates per window
  --keyword-max-per-keyword N
                          Limit keyword matches per keyword per window
  --keyword-score FLOAT   Override score assigned to keyword candidates
  --help                  Show this message
EOF
  exit 1
}

if [ $# -lt 1 ]; then
  usage
fi

CATEGORY=$1
shift

# defaults
CHUNKS=5
BIBLE_INDEX="results/verses_index.parquet"
OUT_DIR="results/app/${CATEGORY}"
SPEECHES_IN="cleaned_data/app/${CATEGORY}/rows_norm.parquet"
MODEL_PATH="models/apb_lgbm_sem.pkl"
THRESHOLD=0.1
NGRAM_MIN=3
NGRAM_MAX=5
TOP_PER_DOC=15
CAND_TOPK=25
BATCH_SIZE=1000
KEYWORD_CAND=true
KEYWORD_MAX_TOTAL=""
KEYWORD_MAX_PER=""
KEYWORD_SCORE=""

# parse optional args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunks)
      CHUNKS="$2"; shift 2;;
    --model-path)
      MODEL_PATH="$2"; shift 2;;
    --threshold)
      THRESHOLD="$2"; shift 2;;
    --ngram-min)
      NGRAM_MIN="$2"; shift 2;;
    --ngram-max)
      NGRAM_MAX="$2"; shift 2;;
    --top-per-doc)
      TOP_PER_DOC="$2"; shift 2;;
    --tfidf-topk)
      CAND_TOPK="$2"; shift 2;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2;;
    --out-dir)
      OUT_DIR="$2"; shift 2;;
    --keyword-candidates)
      KEYWORD_CAND=true; shift;;
    --no-keyword-candidates)
      KEYWORD_CAND=false; shift;;
    --keyword-max-total)
      KEYWORD_MAX_TOTAL="$2"; shift 2;;
    --keyword-max-per-keyword)
      KEYWORD_MAX_PER="$2"; shift 2;;
    --keyword-score)
      KEYWORD_SCORE="$2"; shift 2;;
    --help)
      usage;;
    *)
      echo "Unknown option: $1"; usage;;
  esac
done

mkdir -p "$OUT_DIR"

if [ ! -f "$BIBLE_INDEX" ]; then
  echo "[error] Expected verse index not found at: $BIBLE_INDEX"
  exit 1
fi

if [ -f "$OUT_DIR/speeches_index.parquet" ] && [ -f "$OUT_DIR/windows.parquet" ]; then
  echo "[info] Found speeches_index + windows in $OUT_DIR -> skipping index-speeches"
else
  if [ ! -f "$SPEECHES_IN" ]; then
    echo "[error] Expected cleaned speeches not found at: $SPEECHES_IN"
    echo "Run normalization first or place rows_norm.parquet under cleaned_data/app/${CATEGORY}/"
    exit 2
  fi
  echo "[info] Indexing speeches -> $OUT_DIR/speeches_index.parquet + windows (input: $SPEECHES_IN)"
  python scripts/verse_match_pipeline_app.py index-speeches \
    --speeches_in "$SPEECHES_IN" --out_dir "$OUT_DIR" --window_len 30 --stride 5
fi

CHUNK_DIR="$OUT_DIR/windows_chunks"
CHUNK_OUT_BASE="$OUT_DIR/chunks"
mkdir -p "$CHUNK_DIR" "$CHUNK_OUT_BASE"

echo "[info] Splitting windows into $CHUNKS chunks by doc_id -> $CHUNK_DIR"
python - <<PY
import pandas as pd
from pathlib import Path

win_path = Path("$OUT_DIR") / "windows.parquet"
out_dir = Path("$CHUNK_DIR")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(win_path)
counts = df.groupby("doc_id").size().reset_index(name="n")
total = counts["n"].sum()
target = total / $CHUNKS

chunks = []
current = []
running = 0
for _, row in counts.iterrows():
    if running + row["n"] > target and len(chunks) < ($CHUNKS - 1):
        chunks.append(current)
        current = []
        running = 0
    current.append(row["doc_id"])
    running += row["n"]
if current:
    chunks.append(current)

for i, doc_ids in enumerate(chunks, start=1):
    chunk = df[df["doc_id"].isin(doc_ids)]
    out = out_dir / f"windows_part_{i:02d}.parquet"
    chunk.to_parquet(out, index=False)
    print(out, len(chunk), "docs:", len(doc_ids))
PY

for w in "$CHUNK_DIR"/windows_part_*.parquet; do
  name=$(basename "$w" .parquet)
  out="$CHUNK_OUT_BASE/$name"
  mkdir -p "$out"

  echo "[info] gen-candidates for $name -> $out"
  GEN_CMD=(python scripts/verse_match_pipeline_app.py gen-candidates
    --bible_index "$BIBLE_INDEX"
    --windows "$w"
    --out_dir "$out"
    --ngram_min "$NGRAM_MIN" --ngram_max "$NGRAM_MAX"
    --topk "$CAND_TOPK" --batch_size "$BATCH_SIZE")

  if [ "$KEYWORD_CAND" = true ]; then
    GEN_CMD+=(--keyword-candidates)
    if [ -n "$KEYWORD_MAX_TOTAL" ]; then
      GEN_CMD+=(--keyword-max-total "$KEYWORD_MAX_TOTAL")
    fi
    if [ -n "$KEYWORD_MAX_PER" ]; then
      GEN_CMD+=(--keyword-max-per-keyword "$KEYWORD_MAX_PER")
    fi
    if [ -n "$KEYWORD_SCORE" ]; then
      GEN_CMD+=(--keyword-score "$KEYWORD_SCORE")
    fi
  fi

  "${GEN_CMD[@]}"

  echo "[info] merge-spans for $name -> $out"
  python scripts/verse_match_pipeline_app.py merge-spans \
    --out_dir "$out" \
    --windows "$w" \
    --candidates "$out/candidates.parquet" \
    --bible_index "$BIBLE_INDEX" \
    --speeches_index "$OUT_DIR/speeches_index.parquet" \
    --ngram_min "$NGRAM_MIN" --ngram_max "$NGRAM_MAX" --min_cov 0.45 --min_lcs 0.40 --max_gap 8 --top_per_doc "$TOP_PER_DOC" \
    --model_path "$MODEL_PATH" --threshold "$THRESHOLD"

done

echo "[done] chunked pipeline complete."
