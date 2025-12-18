#!/usr/bin/env python3
"""Send classifier-positive matches through an LLM for final vetting."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import boto3


PROMPT_TEMPLATE = """You verify whether excerpts from presidential eulogies actually reference the quoted Bible verse.
Compare the `snippet_norm` text to the `verse_text` and assess whether the speaker is clearly invoking that scripture.

Judgement categories (choose exactly one):
- TRUE: the snippet clearly paraphrases or directly references the verse.
- FALSE: no meaningful connection to the verse.
- MAYBE: tenuous link or insufficient information to decide.
- Always mark MAYBE when the snippet sounds biblical but appears to cite a different passage than the provided verse or when you cannot confirm an exact correspondence.

Respond with a JSON array of objects. Each object MUST contain:
- "row_index": the input row_index string
- "judgement": one of "TRUE", "FALSE", "MAYBE"
- "confidence": float between 0 and 1

Return ONLY the JSON array enclosed in a fenced block exactly like:
```json
[ ... ]
```
No commentary outside that block. This must be valid JSON (no double quotes, trailing commas, comments, etc.)

{error_hint}

Here are the rows to analyze:
{records}
"""

SELECTED_FIELDS = [
    "row_index",
    "doc_id",
    "title",
    "president",
    "date",
    "snippet_norm",
    "verse_range",
    "verse_text",
    "match_proba",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/app/eulogies/matches_positive.csv",
        help="CSV of classifier positive matches.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/categories",
        help="Directory for TRUE/MAYBE CSV outputs.",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Category slug; used in output filenames.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Rows per LLM request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit for quick smoke tests.",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
        ),
        help="Bedrock model id.",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="If set, also write a CSV containing every verdict.",
    )
    parser.add_argument(
        "--intermediate-name",
        default=None,
        help="Optional filename (relative to output dir) for the intermediate CSV.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="How many times to retry a chunk if the LLM returns invalid JSON (default: 3).",
    )
    parser.add_argument(
        "--debug-llm",
        action="store_true",
        help="If set, print prompts/responses when parsing fails to aid debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_rows(args.input, args.limit))
    if not rows:
        print(f"[warn] No rows loaded from {args.input}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = build_client()
    verdict_rows: List[Dict[str, str]] = []
    for chunk in chunked(rows, args.chunk_size):
        verdicts = request_chunk_with_retries(
            client=client,
            model_id=args.model_id,
            chunk=chunk,
            max_retries=args.max_retries,
            debug=args.debug_llm,
        )
        rows_by_id = {row["row_index"]: row for row in chunk}
        for verdict in verdicts:
            combined = {
                **select_fields(rows_by_id[verdict["row_index"]]),
                "judgement": verdict.get("judgement", ""),
                "reason": verdict.get("reason", ""),
                "confidence": str(verdict.get("confidence", "")),
            }
            verdict_rows.append(combined)

    true_rows = [row for row in verdict_rows if row["judgement"] == "TRUE"]
    maybe_rows = [row for row in verdict_rows if row["judgement"] == "MAYBE"]

    true_path = output_dir / f"{args.category}_llm_true.csv"
    maybe_path = output_dir / f"{args.category}_llm_maybe.csv"
    write_rows(true_path, true_rows)
    write_rows(maybe_path, maybe_rows)
    print(f"[ok] TRUE rows -> {true_path} ({len(true_rows)})")
    print(f"[ok] MAYBE rows -> {maybe_path} ({len(maybe_rows)})")

    if args.save_intermediate:
        fname = (
            args.intermediate_name
            if args.intermediate_name
            else f"{args.category}_llm_all.csv"
        )
        all_path = output_dir / fname
        write_rows(all_path, verdict_rows)
        print(f"[info] Saved all verdicts -> {all_path} ({len(verdict_rows)})")


def read_rows(path: str, limit: int | None) -> Iterator[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            row["row_index"] = str(idx)
            yield row


def select_fields(row: Dict[str, str]) -> Dict[str, str]:
    return {field: row.get(field, "") for field in SELECTED_FIELDS}


def chunked(
    seq: Sequence[Dict[str, str]], size: int
) -> Iterable[List[Dict[str, str]]]:
    for start in range(0, len(seq), size):
        yield list(seq[start : start + size])


def build_client():
    region = os.environ.get("BEDROCK_REGION", "us-east-1")
    session_kwargs = {}
    api_key = os.environ.get("BEDROCK_API_KEY")
    if api_key:
        session_kwargs["aws_access_key_id"] = api_key
    api_secret = os.environ.get("BEDROCK_API_SECRET")
    if api_secret:
        session_kwargs["aws_secret_access_key"] = api_secret
    session_token = os.environ.get("BEDROCK_SESSION_TOKEN")
    if session_token:
        session_kwargs["aws_session_token"] = session_token
    session = boto3.Session(**session_kwargs)
    return session.client("bedrock-runtime", region_name=region)


def invoke_model(client, model_id: str, prompt: str) -> str:
    payload = build_payload(model_id, prompt)
    response = client.invoke_model(modelId=model_id, body=json.dumps(payload))
    body = json.loads(response["body"].read())
    return extract_text(model_id, body)


def request_chunk_with_retries(
    client,
    model_id: str,
    chunk: Sequence[Dict[str, str]],
    max_retries: int,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    last_err: Exception | None = None
    attempts = max(1, max_retries)
    error_hint = ""
    for attempt in range(1, attempts + 1):
        prompt = render_prompt(chunk, error_hint)
        response_text = invoke_model(client, model_id, prompt)
        try:
            verdicts = parse_llm_response(response_text)
            ensure_valid_output(chunk, verdicts)
            return verdicts
        except (json.JSONDecodeError, ValueError) as err:
            last_err = err
            error_hint = str(err)
            if debug:
                _print_debug_failure(prompt, response_text, attempt, err)
            print(
                f"[warn] LLM chunk parse/validation failed on attempt {attempt}/{attempts}: {err}",
                file=sys.stderr,
            )
    raise RuntimeError(
        f"Failed to parse/validate LLM response after {attempts} attempts: {last_err}"
    )


def build_payload(model_id: str, prompt: str) -> Dict[str, Any]:
    max_tokens = int(os.environ.get("BEDROCK_MAX_TOKENS", "2048"))
    temperature = float(os.environ.get("BEDROCK_TEMPERATURE", "0.0"))
    top_p = float(os.environ.get("BEDROCK_TOP_P", "0.9"))

    if model_id.startswith("anthropic."):
        version = os.environ.get("BEDROCK_ANTHROPIC_VERSION", "bedrock-2023-05-31")
        return {
            "anthropic_version": version,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }

    return {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
    }


def extract_text(model_id: str, body: Dict[str, Any]) -> str:
    if model_id.startswith("anthropic."):
        content = body.get("content") or []
        if not content:
            return ""
        first = content[0]
        if isinstance(first, dict) and "text" in first:
            return first.get("text", "")
        inner = first.get("content") if isinstance(first, dict) else None
        if isinstance(inner, list) and inner:
            return inner[0].get("text", "")
        return ""

    results = body.get("results") or []
    if not results:
        return ""
    return results[0].get("outputText", "")


def render_prompt(chunk: Sequence[Dict[str, str]], error_hint: str) -> str:
    hint = ""
    if error_hint:
        hint = (
            "The previous response failed to parse because:\n"
            f"```\n{error_hint}\n```\n"
            "Return valid JSON following the format instructions exactly.\n"
        )
    return PROMPT_TEMPLATE.format(
        records=json.dumps([select_fields(row) for row in chunk], indent=2),
        error_hint=hint,
    )


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_llm_response(raw_text: str) -> Any:
    text = raw_text.strip()
    candidates = [text]
    candidates.extend(match.strip() for match in JSON_FENCE_RE.findall(text))
    last_err: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as err:
            last_err = err
    if last_err:
        raise last_err
    raise ValueError("LLM response was empty")


def ensure_valid_output(
    chunk: Sequence[Dict[str, str]], verdicts: Iterable[Dict[str, str]]
) -> None:
    expected_ids = {row["row_index"] for row in chunk}
    seen = set()
    for verdict in verdicts:
        rid = verdict.get("row_index")
        judgement = verdict.get("judgement")
        if rid not in expected_ids:
            raise ValueError(f"Unexpected row_index in response: {rid}")
        if judgement not in {"TRUE", "FALSE", "MAYBE"}:
            raise ValueError(f"Invalid judgement '{judgement}' for row_index {rid}")
        seen.add(rid)
    if seen != expected_ids:
        missing = expected_ids - seen
        raise ValueError(f"Missing verdicts for row_index values: {sorted(missing)}")


def _print_debug_failure(prompt: str, response: str, attempt: int, err: Exception) -> None:
    def truncate(text: str, limit: int = 800) -> str:
        text = text.strip()
        if len(text) > limit:
            return text[:limit] + "...[truncated]"
        return text

    print(
        f"[debug] Attempt {attempt} failed with error: {err}",
        file=sys.stderr,
    )
    print("[debug] Prompt sent to model:", file=sys.stderr)
    print(truncate(prompt), file=sys.stderr)
    print("[debug] Raw response:", file=sys.stderr)
    print(truncate(response), file=sys.stderr)


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = SELECTED_FIELDS + ["judgement", "reason", "confidence"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
