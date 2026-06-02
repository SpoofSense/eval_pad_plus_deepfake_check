#!/usr/bin/env python3
"""
SpoofSense PAD Liveness + Deepfake Detection evaluation utility.

For each image in an input directory, this script:
  1. Calls the SpoofSense PAD Liveness API   (POST /v3/antispoofing)
  2. Calls the SpoofSense Deepfake Detection API (POST /antispoofing)
  3. Combines the two verdicts: a face is "real" ONLY if BOTH APIs say "real".
  4. Computes a single output score:
       - if exactly one API flags the face as spoof*, use THAT API's prob_real
       - if both APIs agree (both real, or both spoof*), use the AVERAGE prob_real
  5. Writes all per-image results to a CSV.

The utility is dependency-free (Python standard library only).

API reference:
  https://spoofsense.gitbook.io/spoofsense-triton-documentation/api-usage

Both APIs share a response shape:

    {
        "success": true,
        "message": "Process finished successfully",
        "model_output": { "pred_idx": "real", "prob_real": 0.9997 }
    }

Usage:
    python eval_spoofsense.py --input ./selfies --output results.csv \
        --liveness-key  $SPOOFSENSE_LIVENESS_KEY \
        --deepfake-key  $SPOOFSENSE_DEEPFAKE_KEY

Keys may also be supplied via the environment variables
SPOOFSENSE_LIVENESS_KEY and SPOOFSENSE_DEEPFAKE_KEY.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# The two APIs live on different hosts. Each may be overridden via CLI flag
# (--liveness-url / --deepfake-url) or the matching environment variable.
DEFAULT_LIVENESS_URL = (
    "https://z3jwq0rjyj.execute-api.ap-south-1.amazonaws.com/prod/v3/antispoofing"
)  # SpoofSense PAD Liveness API (FaceLive V3)
DEFAULT_DEEPFAKE_URL = (
    "https://47ryr1ufs9.execute-api.ap-south-1.amazonaws.com/prod/antispoofing"
)  # SpoofSense Deepfake Detection API

# Image extensions we will try to evaluate.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# A prediction is considered genuine only when pred_idx equals exactly "real".
# Anything else ("spoof", "spoof_replay", "spoof_print", ...) is treated as spoof.
REAL_LABEL = "real"


# --------------------------------------------------------------------------- #
# API response model
# --------------------------------------------------------------------------- #

@dataclass
class ApiResult:
    """Outcome of a single API call for a single image."""
    pred_idx: Optional[str] = None      # e.g. "real" / "spoof" / "spoof_replay"
    prob_real: Optional[float] = None   # liveness probability in [0, 1]
    error: Optional[str] = None         # populated when the call failed

    @property
    def ok(self) -> bool:
        return self.error is None and self.pred_idx is not None

    @property
    def is_real(self) -> bool:
        return self.ok and self.pred_idx.strip().lower() == REAL_LABEL

    @property
    def is_spoof(self) -> bool:
        return self.ok and not self.is_real


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def encode_image(path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def call_api(url: str, api_key: str, image_b64: str,
             timeout: float, retries: int) -> ApiResult:
    """
    POST a base64 image to a SpoofSense endpoint and parse the response.

    Request body:  {"data": "<base64-image>"}
    Auth header:   x-api-key: <api_key>
    """
    body = json.dumps({"data": image_b64}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }

    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, data=body, headers=headers, method="POST"
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return parse_response(payload)

        except urllib.error.HTTPError as exc:
            # The API often returns business errors with a non-200 status and a
            # JSON body; surface that body when present.
            detail = _read_http_error(exc)
            last_error = f"HTTP {exc.code}: {detail}"
            # 4xx (bad key, payload, too large) won't be fixed by retrying.
            if 400 <= exc.code < 500:
                break

        except urllib.error.URLError as exc:
            last_error = f"network error: {exc.reason}"

        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"invalid response: {exc}"

        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))  # simple linear backoff

    return ApiResult(error=last_error)


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    """Best-effort extraction of a useful message from an HTTPError body."""
    try:
        raw = exc.read().decode("utf-8")
    except Exception:
        return exc.reason or "request failed"
    try:
        data = json.loads(raw)
        return str(data.get("message") or data.get("error") or data)
    except json.JSONDecodeError:
        return raw.strip() or (exc.reason or "request failed")


def parse_response(payload: dict) -> ApiResult:
    """Turn a decoded JSON response into an ApiResult."""
    if not isinstance(payload, dict):
        return ApiResult(error="unexpected response type")

    # Some error responses carry success=false and an explanatory message
    # (FACE_NOT_DETECTED, INVALID_IMAGE_PAYLOAD, FACE_TOO_SMALL, ...).
    model_output = payload.get("model_output")
    if not isinstance(model_output, dict):
        message = payload.get("message") or payload.get("error") or "no model_output"
        return ApiResult(error=str(message))

    pred_idx = model_output.get("pred_idx")
    prob_real = model_output.get("prob_real")
    if pred_idx is None:
        return ApiResult(error="response missing pred_idx")

    try:
        prob_real = float(prob_real) if prob_real is not None else None
    except (TypeError, ValueError):
        prob_real = None

    return ApiResult(pred_idx=str(pred_idx), prob_real=prob_real)


# --------------------------------------------------------------------------- #
# Combination logic
# --------------------------------------------------------------------------- #

@dataclass
class CombinedResult:
    final_verdict: str               # "real", "spoof", or "error"
    output_score: Optional[float]    # combined prob_real, or None on error
    status: str                      # "ok" or an error description


def combine(liveness: ApiResult, deepfake: ApiResult) -> CombinedResult:
    """
    Apply the agreement rules:
      - real  only if BOTH APIs say "real"
      - score: if exactly one API says spoof*, use that API's prob_real;
               if both agree (both real or both spoof*), use the average.
    """
    if not liveness.ok or not deepfake.ok:
        problems = []
        if not liveness.ok:
            problems.append(f"liveness: {liveness.error}")
        if not deepfake.ok:
            problems.append(f"deepfake: {deepfake.error}")
        return CombinedResult("error", None, "; ".join(problems))

    final_verdict = "real" if (liveness.is_real and deepfake.is_real) else "spoof"

    spoofers = [r for r in (liveness, deepfake) if r.is_spoof]
    if len(spoofers) == 1:
        # Exactly one API flagged a spoof -> trust its prob_real.
        output_score = spoofers[0].prob_real
    else:
        # Both agree (both real, or both spoof*) -> average the two scores.
        output_score = _average(liveness.prob_real, deepfake.prob_real)

    return CombinedResult(final_verdict, output_score, "ok")


def _average(*values: Optional[float]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


# --------------------------------------------------------------------------- #
# Per-image evaluation
# --------------------------------------------------------------------------- #

@dataclass
class Row:
    filename: str
    image_path: str
    liveness_pred_idx: str
    liveness_prob_real: str
    deepfake_pred_idx: str
    deepfake_prob_real: str
    final_verdict: str
    output_score: str
    status: str


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def evaluate_image(path: Path, args) -> Row:
    """Run both APIs on one image and build a CSV row."""
    try:
        image_b64 = encode_image(path)
    except OSError as exc:
        return Row(
            filename=path.name, image_path=str(path),
            liveness_pred_idx="", liveness_prob_real="",
            deepfake_pred_idx="", deepfake_prob_real="",
            final_verdict="error", output_score="",
            status=f"could not read file: {exc}",
        )

    # Liveness first, then deepfake (per the required ordering).
    liveness = call_api(
        args.liveness_url, args.liveness_key,
        image_b64, args.timeout, args.retries,
    )
    deepfake = call_api(
        args.deepfake_url, args.deepfake_key,
        image_b64, args.timeout, args.retries,
    )

    combined = combine(liveness, deepfake)

    return Row(
        filename=path.name,
        image_path=str(path),
        liveness_pred_idx=liveness.pred_idx or "",
        liveness_prob_real=_fmt(liveness.prob_real),
        deepfake_pred_idx=deepfake.pred_idx or "",
        deepfake_prob_real=_fmt(deepfake.prob_real),
        final_verdict=combined.final_verdict,
        output_score=_fmt(combined.output_score),
        status=combined.status,
    )


# --------------------------------------------------------------------------- #
# Discovery / IO
# --------------------------------------------------------------------------- #

def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    walker = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images = sorted(
        p for p in walker
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images


CSV_HEADER = [
    "filename", "image_path",
    "liveness_pred_idx", "liveness_prob_real",
    "deepfake_pred_idx", "deepfake_prob_real",
    "final_verdict", "output_score", "status",
]


def write_csv(rows: list[Row], output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for r in rows:
            writer.writerow([
                r.filename, r.image_path,
                r.liveness_pred_idx, r.liveness_prob_real,
                r.deepfake_pred_idx, r.deepfake_prob_real,
                r.final_verdict, r.output_score, r.status,
            ])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate images against SpoofSense PAD Liveness + "
                    "Deepfake Detection APIs and write a combined CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Directory of images to evaluate (any mix of real/spoof).",
    )
    parser.add_argument(
        "-o", "--output", default=Path("spoofsense_eval_results.csv"), type=Path,
        help="Path to the output CSV.",
    )
    parser.add_argument(
        "--liveness-key", default=os.environ.get("SPOOFSENSE_LIVENESS_KEY"),
        help="Sandbox API key for the PAD Liveness endpoint "
             "(or set SPOOFSENSE_LIVENESS_KEY).",
    )
    parser.add_argument(
        "--deepfake-key", default=os.environ.get("SPOOFSENSE_DEEPFAKE_KEY"),
        help="Sandbox API key for the Deepfake Detection endpoint "
             "(or set SPOOFSENSE_DEEPFAKE_KEY).",
    )
    parser.add_argument(
        "--liveness-url",
        default=os.environ.get("SPOOFSENSE_LIVENESS_URL", DEFAULT_LIVENESS_URL),
        help="Full URL for the PAD Liveness endpoint.",
    )
    parser.add_argument(
        "--deepfake-url",
        default=os.environ.get("SPOOFSENSE_DEEPFAKE_URL", DEFAULT_DEEPFAKE_URL),
        help="Full URL for the Deepfake Detection endpoint.",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Recurse into subdirectories when looking for images.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of images to process concurrently.",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries", type=int, default=2,
        help="Retries per request on transient (network/5xx) errors.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Validate inputs early with clear messages.
    missing_keys = []
    if not args.liveness_key:
        missing_keys.append("--liveness-key / SPOOFSENSE_LIVENESS_KEY")
    if not args.deepfake_key:
        missing_keys.append("--deepfake-key / SPOOFSENSE_DEEPFAKE_KEY")
    if missing_keys:
        print("ERROR: missing required API key(s):", file=sys.stderr)
        for m in missing_keys:
            print(f"  - {m}", file=sys.stderr)
        return 2

    if not args.input.is_dir():
        print(f"ERROR: input is not a directory: {args.input}", file=sys.stderr)
        return 2

    images = find_images(args.input, args.recursive)
    if not images:
        print(f"No images found in {args.input} "
              f"(extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}).",
              file=sys.stderr)
        return 1

    print(f"Found {len(images)} image(s). Evaluating against:")
    print(f"  Liveness : {args.liveness_url}")
    print(f"  Deepfake : {args.deepfake_url}")
    print()

    rows: list[Row] = []
    workers = max(1, args.workers)
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_image, img, args): img for img in images}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            done += 1
            verdict = row.final_verdict.upper()
            score = row.output_score or "-"
            note = "" if row.status == "ok" else f"  [{row.status}]"
            print(f"[{done}/{len(images)}] {row.filename}: "
                  f"{verdict} (score={score}){note}")

    # Keep CSV order stable (by filename) regardless of completion order.
    rows.sort(key=lambda r: r.filename)
    write_csv(rows, args.output)

    summary = _summarize(rows)
    print()
    print("Summary:")
    print(f"  real   : {summary['real']}")
    print(f"  spoof  : {summary['spoof']}")
    print(f"  error  : {summary['error']}")
    print(f"\nResults written to {args.output}")
    return 0


def _summarize(rows: list[Row]) -> dict:
    counts = {"real": 0, "spoof": 0, "error": 0}
    for r in rows:
        counts[r.final_verdict] = counts.get(r.final_verdict, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
