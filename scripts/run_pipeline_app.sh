#!/usr/bin/env bash
# scripts/run_pipeline_app.sh
# Convenience wrapper to run the full pipeline for a single APP category
# Usage:
#   bash scripts/run_pipeline_app.sh <category> [--model-path PATH] [--write-intermediate] [--ngram-min N] [--ngram-max N] [--top-per-doc N]
# Example:
#   bash scripts/run_pipeline_app.sh eulogies --model-path models/apb_lgbm_sem.pkl

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 <category> [options]

Options:
  --model-path PATH        Path to saved BibleMatchClassifier (default: models/apb_lgbm_sem.pkl)
  --threshold FLOAT        Decision threshold for classifier (default: 0.01)
  --write-intermediate     If set, write matches.parquet and matches_preview.csv in addition to matches_scored.parquet
  --write-scored           Persist matches_scored.parquet (default: off)
  --ngram-min N            ngram min for gen-candidates (default: 3)
  --ngram-max N            ngram max for gen-candidates (default: 5)
  --top-per-doc N          how many merged hits to keep per doc for preview (default: 15)
  --tfidf-topk N           TF-IDF candidates to keep per window (default: 25)
  --help                   Show this message

Example:
  bash $0 eulogies --model-path models/apb_lgbm_sem.pkl
EOF
  exit 1
}

if [ $# -lt 1 ]; then
  usage
fi

CATEGORY=$1
shift

# defaults
BIBLE_INDEX="results/verses_index.parquet"
OUT_DIR="results/app/${CATEGORY}"
SPEECHES_IN="cleaned_data/app/${CATEGORY}/rows_norm.parquet"
MODEL_PATH="models/apb_lgbm_sem.pkl"
THRESHOLD=0.005
NGRAM_MIN=3
NGRAM_MAX=5
TOP_PER_DOC=15
CAND_TOPK=25
WRITE_INTERMEDIATE=false
WRITE_SCORED=false

# parse optional args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"; shift 2;;
    --threshold)
      THRESHOLD="$2"; shift 2;;
    --write-intermediate)
      WRITE_INTERMEDIATE=true; shift;;
    --write-scored)
      WRITE_SCORED=true; shift;;
    --require-transcript)
      REQUIRE_TRANSCRIPT=true; shift;;
    --items-per-page)
      ITEMS_PER_PAGE="$2"; shift 2;;
    --start-year)
      START_YEAR="$2"; shift 2;;
    --end-year)
      END_YEAR="$2"; shift 2;;
    --ngram-min)
      NGRAM_MIN="$2"; shift 2;;
    --ngram-max)
      NGRAM_MAX="$2"; shift 2;;
    --top-per-doc)
      TOP_PER_DOC="$2"; shift 2;;
    --tfidf-topk)
      CAND_TOPK="$2"; shift 2;;
    --help)
      usage;;
    *)
      echo "Unknown option: $1"; usage;;
  esac
done

mkdir -p "$OUT_DIR"

# Use a pre-built global verse index (do not create a per-results verses_index)
if [ -f "$BIBLE_INDEX" ]; then
  echo "[info] Using existing verse index at $BIBLE_INDEX"
else
  echo "[error] Expected verse index not found at: $BIBLE_INDEX"
  echo "Please create or place verses_index.parquet at $BIBLE_INDEX or modify the script to point elsewhere."
  exit 1
fi

# Determine speech input: prefer an existing speeches_index in results, else try cleaned_data, then data, else scrape+normalize
if [ -f "$OUT_DIR/speeches_index.parquet" ] && [ -f "$OUT_DIR/windows.parquet" ]; then
  echo "[info] Found existing speeches_index and windows in $OUT_DIR -> skipping index-speeches"
else
  # If cleaned normalized data exists, use it
  if [ -f "cleaned_data/app/${CATEGORY}/rows_norm.parquet" ]; then
    SPEECHES_IN="cleaned_data/app/${CATEGORY}/rows_norm.parquet"
    echo "[info] Using existing cleaned speeches: $SPEECHES_IN"
  else
    # If raw CSV exists under data/, prefer it
    if [ -f "data/app/${CATEGORY}/rows.csv" ]; then
      SRC_CSV="data/app/${CATEGORY}/rows.csv"
    elif [ -f "data/app/${CATEGORY}/rows.parquet" ]; then
      # convert parquet -> csv for normalization script
      echo "[info] Converting data/app/${CATEGORY}/rows.parquet -> CSV for normalization"
      mkdir -p "data/app/${CATEGORY}"
      python3 - <<PY
import pandas as pd
df = pd.read_parquet('data/app/${CATEGORY}/rows.parquet')
df.to_csv('data/app/${CATEGORY}/rows.csv', index=False)
print('WROTE data/app/${CATEGORY}/rows.csv')
PY
      SRC_CSV="data/app/${CATEGORY}/rows.csv"
    else
      # No raw data found; run scraper to data/app/<category>
      echo "[info] No raw data found under data/; running scraper to populate data/app/${CATEGORY}"
      mkdir -p "data/app/${CATEGORY}"
      SCRAPE_CMD=(python3 scripts/scrape_app_category.py --slug "${CATEGORY}" --out-dir "data/app/${CATEGORY}")
      if [ "${REQUIRE_TRANSCRIPT:-false}" = true ]; then
        SCRAPE_CMD+=(--require-transcript)
      fi
      if [ -n "${ITEMS_PER_PAGE:-}" ]; then
        SCRAPE_CMD+=(--items-per-page "${ITEMS_PER_PAGE}")
      fi
      if [ -n "${START_YEAR:-}" ]; then
        SCRAPE_CMD+=(--start-year "${START_YEAR}")
      fi
      if [ -n "${END_YEAR:-}" ]; then
        SCRAPE_CMD+=(--end-year "${END_YEAR}")
      fi
      echo "[debug] running: ${SCRAPE_CMD[*]}"
      "${SCRAPE_CMD[@]}"
      if [ -f "data/app/${CATEGORY}/rows.csv" ]; then
        SRC_CSV="data/app/${CATEGORY}/rows.csv"
      elif [ -f "data/app/${CATEGORY}/rows.parquet" ]; then
        # convert parquet -> csv
        python3 - <<PY
import pandas as pd
df = pd.read_parquet('data/app/${CATEGORY}/rows.parquet')
df.to_csv('data/app/${CATEGORY}/rows.csv', index=False)
print('WROTE data/app/${CATEGORY}/rows.csv')
PY
        SRC_CSV="data/app/${CATEGORY}/rows.csv"
      else
        echo "[error] Scraper did not produce rows.csv or rows.parquet in data/app/${CATEGORY}"
        exit 3
      fi
    fi

    # Run normalization to produce cleaned_data/.../rows_norm.parquet
    echo "[info] Normalizing transcripts -> cleaned_data/app/${CATEGORY}/rows_norm.parquet (input: ${SRC_CSV})"
    python3 scripts/normalize_app.py --in "${SRC_CSV}"
    # normalize_app.py writes cleaned_data/.../<stem>_norm.parquet; infer output path
    # The normalize script names output based on input; we'll assume out path ends with rows_norm.parquet
    if [ -f "cleaned_data/app/${CATEGORY}/rows_norm.parquet" ]; then
      SPEECHES_IN="cleaned_data/app/${CATEGORY}/rows_norm.parquet"
    else
      # attempt to find any *_norm.parquet under cleaned_data/app/<category>
      NORM_PQ=$(ls cleaned_data/app/${CATEGORY}/*_norm.parquet 2>/dev/null || true)
      if [ -n "$NORM_PQ" ]; then
        SPEECHES_IN="$NORM_PQ"
      else
        echo "[error] Normalization did not produce cleaned_data/app/${CATEGORY}/rows_norm.parquet"
        exit 4
      fi
    fi
  fi

  echo "[info] 2/4 Indexing speeches -> $OUT_DIR/speeches_index.parquet + windows (input: $SPEECHES_IN)"
  python3 scripts/verse_match_pipeline_app.py index-speeches \
    --speeches_in "$SPEECHES_IN" --out_dir "$OUT_DIR" --window_len 30 --stride 5
fi

echo "[info] 3/4 Generating TF-IDF candidates -> $OUT_DIR/candidates.parquet"
python3 scripts/verse_match_pipeline_app.py gen-candidates \
  --bible_index "$BIBLE_INDEX" \
  --windows "$OUT_DIR/windows.parquet" \
  --out_dir "$OUT_DIR" --ngram_min $NGRAM_MIN --ngram_max $NGRAM_MAX --topk $CAND_TOPK --batch_size 1000

# Merge spans and (optionally) score in one step
echo "[info] 4/4 Merging spans and scoring -> matches_scored.parquet"
CMD=(python3 scripts/verse_match_pipeline_app.py merge-spans --out_dir "$OUT_DIR" \
  --ngram_min $NGRAM_MIN --ngram_max $NGRAM_MAX --min_cov 0.45 --min_lcs 0.40 --max_gap 8 --top_per_doc $TOP_PER_DOC \
  --bible_index "$BIBLE_INDEX" \
  --model_path "$MODEL_PATH" --threshold $THRESHOLD)

if [ "$WRITE_INTERMEDIATE" = true ]; then
  CMD+=(--write_intermediate)
fi

if [ "$WRITE_SCORED" = true ]; then
  CMD+=(--write_scored)
fi

# join and run
echo "[debug] running: ${CMD[*]}"
"${CMD[@]}"

echo "[done] pipeline complete. Positive matches at: $OUT_DIR/matches_positive.csv"
if [ "$WRITE_SCORED" = true ]; then
  echo "[info] Full scored matches at: $OUT_DIR/matches_scored.parquet"
fi

exit 0
