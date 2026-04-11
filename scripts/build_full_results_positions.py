#!/usr/bin/env python3
from __future__ import annotations

import csv
from difflib import SequenceMatcher
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL_RESULTS_PATH = ROOT / "cleaned-results" / "full-results.csv"
OUTPUT_PATH = ROOT / "cleaned-results" / "full-results-with-positions.csv"


def _read_full_results(path: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    repaired = 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_len = len(header)

        for line_no, row in enumerate(reader, start=2):
            if len(row) == expected_len + 1 and row[8] == "":
                # One row has an extra empty field before match_proba.
                row = row[:8] + row[9:]
                repaired += 1

            if len(row) != expected_len:
                raise ValueError(
                    f"Unexpected column count in {path} at line {line_no}: "
                    f"expected {expected_len}, found {len(row)}"
                )

            rows.append(dict(zip(header, row)))

    df = pd.DataFrame(rows)
    print(f"[ok] full-results rows: {len(df)} (repaired rows: {repaired})")
    return df


def _load_match_positions(root: Path) -> pd.DataFrame:
    frames = []
    app_root = root / "results" / "app"
    for category_dir in sorted(app_root.glob("*")):
        direct_path = category_dir / "matches_positive.csv"
        if direct_path.exists():
            df = pd.read_csv(
                direct_path,
                usecols=[
                    "doc_id",
                    "snippet_norm",
                    "verse_range",
                    "start_token",
                    "end_token",
                ],
            )
            df["source_category"] = category_dir.name
            df["source_chunk"] = ""
            frames.append(df)

        chunks_dir = category_dir / "chunks"
        for chunk_dir in sorted(chunks_dir.glob("windows_part_*")):
            path = chunk_dir / "matches_positive.csv"
            df = pd.read_csv(
                path,
                usecols=[
                    "doc_id",
                    "snippet_norm",
                    "verse_range",
                    "start_token",
                    "end_token",
                ],
            )
            df["source_category"] = category_dir.name
            df["source_chunk"] = chunk_dir.name
            frames.append(df)

    all_matches = pd.concat(frames, ignore_index=True)
    all_matches["doc_id"] = all_matches["doc_id"].astype(str)

    dupes = all_matches.duplicated(["doc_id", "snippet_norm", "verse_range"])
    if dupes.any():
        raise ValueError("Join key is not unique in matches_positive.csv inputs")

    snippet_positions = (
        all_matches.groupby(["doc_id", "snippet_norm"], as_index=False)
        .agg(
            start_token=("start_token", "first"),
            end_token=("end_token", "first"),
            source_category=("source_category", "first"),
            source_chunk=("source_chunk", "first"),
        )
    )

    pos_dupes = snippet_positions.duplicated(["doc_id", "snippet_norm"])
    if pos_dupes.any():
        raise ValueError("Snippet-level position key is not unique")

    print(
        f"[ok] positional match rows: {len(all_matches)} "
        f"(unique snippet positions: {len(snippet_positions)})"
    )
    return snippet_positions


def _load_speech_metadata(root: Path) -> pd.DataFrame:
    frames = []
    for category_dir in sorted((root / "cleaned_data" / "app").glob("*")):
        path = category_dir / "rows_norm.csv"
        if not path.exists():
            continue
        df = pd.read_csv(
            path,
            usecols=[
                "doc_id",
                "title",
                "url",
                "location",
                "word_count",
                "token_count_norm",
                "transcript_norm",
            ],
        )
        df["source_category"] = category_dir.name
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["doc_id"] = out["doc_id"].astype(str)
    out = out.drop_duplicates(subset=["doc_id"], keep="first")
    print(f"[ok] speech metadata rows: {len(out)}")
    return out


def _find_sublist_start(haystack: list[str], needle: list[str]) -> int | None:
    if not haystack or not needle or len(needle) > len(haystack):
        return None
    last = len(haystack) - len(needle) + 1
    for i in range(last):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def _canonicalize_with_map(text: str) -> tuple[list[str], list[int]]:
    canonical_tokens: list[str] = []
    original_map: list[int] = []
    for original_idx, token in enumerate(str(text).split()):
        pieces = re.findall(r"[a-z0-9']+", token.lower())
        for piece in pieces:
            canonical_tokens.append(piece)
            original_map.append(original_idx)
    return canonical_tokens, original_map


def _recover_position_from_transcript(transcript: str, snippet: str) -> tuple[int, int] | None:
    raw_doc_tokens = str(transcript).split()
    doc_tokens, doc_map = _canonicalize_with_map(transcript)
    snippet_tokens, _ = _canonicalize_with_map(snippet)
    if not doc_tokens or not snippet_tokens or len(snippet_tokens) > len(doc_tokens):
        return None

    direct = _find_sublist_start(doc_tokens, snippet_tokens)
    if direct is not None:
        start = doc_map[direct]
        end = doc_map[direct + len(snippet_tokens) - 1] + 1
        return start, end

    n = min(5, len(snippet_tokens))
    while n >= 3:
        anchors: dict[tuple[str, ...], list[int]] = {}
        for i in range(len(doc_tokens) - n + 1):
            anchors.setdefault(tuple(doc_tokens[i : i + n]), []).append(i)

        candidate_starts: dict[int, int] = {}
        for j in range(len(snippet_tokens) - n + 1):
            key = tuple(snippet_tokens[j : j + n])
            for pos in anchors.get(key, []):
                start = pos - j
                if start < 0 or start + len(snippet_tokens) > len(doc_tokens):
                    continue
                candidate_starts[start] = candidate_starts.get(start, 0) + 1

        if candidate_starts:
            best_span: tuple[float, int, int, int, int] | None = None
            min_len = max(1, len(snippet_tokens) - 5)
            max_len = min(len(doc_tokens), len(snippet_tokens) + 10)
            for cand_start, anchor_hits in candidate_starts.items():
                for window_len in range(min_len, max_len + 1):
                    cand_end = cand_start + window_len
                    if cand_end > len(doc_tokens):
                        break
                    window = doc_tokens[cand_start:cand_end]
                    sim = SequenceMatcher(a=snippet_tokens, b=window).ratio()
                    score = (
                        sim,
                        anchor_hits,
                        -abs(window_len - len(snippet_tokens)),
                        cand_start,
                        cand_end,
                    )
                    if best_span is None or score > best_span:
                        best_span = score

            if best_span and best_span[0] >= 0.7:
                _, _, _, best_start, best_end = best_span
                start = doc_map[best_start]
                end = doc_map[best_end - 1] + 1
                if 0 <= start < end <= len(raw_doc_tokens):
                    return start, end
        n -= 1

    return None


def main() -> None:
    full_results = _read_full_results(FULL_RESULTS_PATH)
    full_results["doc_id"] = full_results["doc_id"].astype(str)

    match_positions = _load_match_positions(ROOT)
    speeches = _load_speech_metadata(ROOT)

    merged = full_results.merge(
        match_positions,
        on=["doc_id", "snippet_norm"],
        how="left",
    )
    merged = merged.merge(speeches, on="doc_id", how="left")
    if "title_x" in merged.columns:
        merged = merged.rename(columns={"title_x": "title"})
    if "title_y" in merged.columns:
        merged = merged.rename(columns={"title_y": "source_title"})
    if "source_category_x" in merged.columns and "source_category_y" in merged.columns:
        merged["source_category"] = merged["source_category_x"].combine_first(
            merged["source_category_y"]
        )
    elif "source_category_x" in merged.columns:
        merged = merged.rename(columns={"source_category_x": "source_category"})
    elif "source_category_y" in merged.columns:
        merged = merged.rename(columns={"source_category_y": "source_category"})
    if merged["token_count_norm"].isna().any():
        missing = int(merged["token_count_norm"].isna().sum())
        raise ValueError(f"Missing token_count_norm for {missing} rows")

    missing_mask = merged["start_token"].isna()
    recovered = 0
    if missing_mask.any():
        for idx in merged.index[missing_mask]:
            rec = _recover_position_from_transcript(
                merged.at[idx, "transcript_norm"], merged.at[idx, "snippet_norm"]
            )
            if rec is None:
                continue
            start_token, end_token = rec
            merged.at[idx, "start_token"] = start_token
            merged.at[idx, "end_token"] = end_token
            recovered += 1

    if merged["start_token"].isna().any():
        missing_rows = merged.loc[merged["start_token"].isna(), ["doc_id", "title", "verse_range"]]
        raise ValueError(
            f"Missing positional data for {len(missing_rows)} rows after transcript fallback: "
            f"{missing_rows.head(10).to_dict(orient='records')}"
        )
    print(f"[ok] transcript fallback recovered rows: {recovered}")

    merged["start_token"] = merged["start_token"].astype(int)
    merged["end_token"] = merged["end_token"].astype(int)
    merged["token_count_norm"] = merged["token_count_norm"].astype(int)
    merged["word_count"] = pd.to_numeric(merged["word_count"], errors="coerce").astype("Int64")
    merged["window_token_count"] = merged["end_token"] - merged["start_token"]
    merged["window_mid_token"] = (merged["start_token"] + merged["end_token"]) / 2

    merged["start_pct_through_speech"] = merged["start_token"] / merged["token_count_norm"]
    merged["end_pct_through_speech"] = merged["end_token"] / merged["token_count_norm"]
    merged["mid_pct_through_speech"] = merged["window_mid_token"] / merged["token_count_norm"]

    output_cols = [
        "row_index",
        "doc_id",
        "title",
        "date",
        "president",
        "source_category",
        "location",
        "url",
        "verse_range",
        "verse_text",
        "snippet_norm",
        "judgement",
        "confidence",
        "match_proba",
        "start_token",
        "end_token",
        "window_token_count",
        "window_mid_token",
        "token_count_norm",
        "word_count",
        "start_pct_through_speech",
        "mid_pct_through_speech",
        "end_pct_through_speech",
        "source_chunk",
    ]

    merged = merged[output_cols].sort_values(
        ["date", "president", "title", "start_token", "verse_range"]
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"[ok] wrote {len(merged)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
