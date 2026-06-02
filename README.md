# SpoofSense PAD Liveness + Deepfake Detection — Eval Utility

A small, dependency-free Python utility that lets you evaluate a folder of
selfies (genuine or deepfake) against **both** SpoofSense APIs and get a single,
combined verdict per image in a CSV.

## Requirements

- Python 3.8+
- No third-party packages (uses only the standard library).
- Sandbox API keys for **both** endpoints.

## Setup

Provide your sandbox keys via environment variables (recommended):

```bash
export SPOOFSENSE_LIVENESS_KEY="your-liveness-sandbox-key"
export SPOOFSENSE_DEEPFAKE_KEY="your-deepfake-sandbox-key"
```

…or pass them on the command line with `--liveness-key` / `--deepfake-key`.

## Usage

```bash
python eval_spoofsense.py --input ./selfies --output results.csv
```

The input directory can contain **any** mix of images — only real, only spoof,
or a blend. Nothing is assumed about the contents.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --input` | *(required)* | Directory of images to evaluate. |
| `-o, --output` | `spoofsense_eval_results.csv` | Output CSV path. |
| `--liveness-key` | `$SPOOFSENSE_LIVENESS_KEY` | Sandbox key for the PAD Liveness endpoint. |
| `--deepfake-key` | `$SPOOFSENSE_DEEPFAKE_KEY` | Sandbox key for the Deepfake Detection endpoint. |
| `-r, --recursive` | off | Recurse into subdirectories. |
| `--workers` | `4` | Images processed concurrently. |
| `--timeout` | `30` | Per-request timeout (seconds). |
| `--retries` | `2` | Retries on transient (network/5xx) errors. |

Supported image types: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

## Output CSV

| Column | Meaning |
|--------|---------|
| `filename` | Image file name. |
| `image_path` | Full path to the image. |
| `liveness_pred_idx` | `pred_idx` from the PAD Liveness API. |
| `liveness_prob_real` | `prob_real` from the PAD Liveness API. |
| `deepfake_pred_idx` | `pred_idx` from the Deepfake Detection API. |
| `deepfake_prob_real` | `prob_real` from the Deepfake Detection API. |
| `final_verdict` | `real`, `spoof`, or `error`. |
| `output_score` | Combined `prob_real` (see scoring rules above). |
| `status` | `ok`, or the error message for that image. |

Images that fail (e.g. `FACE_NOT_DETECTED`, `FACE_TOO_SMALL`, bad key, network
issues) are reported with `final_verdict = error` and the reason in `status`, so
a single bad image never stops the whole run.

## Example

```bash
$ python eval_spoofsense.py -i ./selfies -o results.csv
Found 3 image(s). Evaluating against:
  Liveness : https://.../prod/v3/antispoofing
  Deepfake : https://.../prod/antispoofing

[1/3] genuine_01.jpg: REAL (score=0.987654)
[2/3] replay_02.jpg: SPOOF (score=0.142000)
[3/3] deepfake_03.png: SPOOF (score=0.090000)

Summary:
  real   : 1
  spoof  : 2
  error  : 0

Results written to results.csv
```
