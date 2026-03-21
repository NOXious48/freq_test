# scripts/generate_submission.py
# Deep Sentinel — FaceForensics++ Benchmark Submission Generator
#
# Converts batch_run.py JSON output into the submission format required by:
#   https://kaldir.vc.in.tum.de/faceforensics_benchmark
#
# Submission format (from benchmark documentation):
#   A single .json file containing a dictionary of predicted labels.
#   Label values must be exactly "fake" or "real" (strings).
#   Example: {"0000.png": "fake", "0001.png": "real", ...}
#
# Submit by zipping the JSON and uploading to the benchmark site.
#
# Usage:
#   conda run -n deep_sentinel python scripts/generate_submission.py
#   conda run -n deep_sentinel python scripts/generate_submission.py reports/batch_results_TIMESTAMP.json
#   conda run -n deep_sentinel python scripts/generate_submission.py reports/batch_results_TIMESTAMP.json 0.50

import json
import sys
import zipfile
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

# Threshold for FAKE label.
# Images with anomaly_score >= threshold → "fake"
# Images with anomaly_score <  threshold → "real"
# Default 0.50 — change via command line arg or edit here.
DEFAULT_THRESHOLD = 0.50

# ── Main ──────────────────────────────────────────────────────────────────────

def generate_submission(batch_json_path: str, threshold: float = DEFAULT_THRESHOLD):

    batch_path = Path(batch_json_path)
    if not batch_path.exists():
        print(f"[ERROR] File not found: {batch_path}")
        sys.exit(1)

    with open(batch_path) as f:
        data = json.load(f)

    results = data["results"]

    # Only include .png files (the benchmark images 0000.png–0999.png)
    # Exclude test_*.jpg files which are local test images, not benchmark images
    benchmark_results = [
        r for r in results
        if r["filename"].endswith(".png") and r.get("valid", False)
    ]

    if not benchmark_results:
        print("[ERROR] No valid .png benchmark images found in batch results.")
        print("        Make sure you ran batch_run.py on the FF++ benchmark images directory.")
        sys.exit(1)

    # Build submission dict
    predictions = {}
    fake_count = 0
    real_count = 0

    for r in benchmark_results:
        label = "fake" if r["anomaly_score"] >= threshold else "real"
        predictions[r["filename"]] = label
        if label == "fake":
            fake_count += 1
        else:
            real_count += 1

    # Output paths
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir  = batch_path.parent
    json_name   = f"submission_{ts}.json"
    zip_name    = f"submission_{ts}.zip"
    json_path   = output_dir / json_name
    zip_path    = output_dir / zip_name

    # Write JSON
    with open(json_path, "w") as f:
        json.dump(predictions, f, indent=2)

    # Wrap in zip (required by benchmark)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname=json_name)

    # Print summary
    total = len(predictions)
    print(f"\n── FaceForensics++ Benchmark Submission ──")
    print(f"  Source batch : {batch_path.name}")
    print(f"  Images       : {total}")
    print(f"  Threshold    : anomaly_score >= {threshold} → 'fake'")
    print(f"  Fake         : {fake_count}/{total}  ({100*fake_count/total:.1f}%)")
    print(f"  Real         : {real_count}/{total}  ({100*real_count/total:.1f}%)")
    print(f"\n  JSON         : {json_path}")
    print(f"  ZIP (submit) : {zip_path}")
    print(f"\nUpload {zip_name} at:")
    print(f"  https://kaldir.vc.in.tum.de/faceforensics_benchmark")

    # Preview first 5 predictions
    print(f"\nFirst 5 predictions:")
    for fname, label in list(predictions.items())[:5]:
        score = next(r["anomaly_score"] for r in benchmark_results if r["filename"] == fname)
        print(f"  {fname}  score={score:.4f}  → {label}")

    return predictions


if __name__ == "__main__":
    # Args: [batch_json_path] [threshold]
    batch_json = sys.argv[1] if len(sys.argv) > 1 else None
    threshold  = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD

    # Auto-find latest batch results if no path given
    if batch_json is None:
        reports_dir = Path("reports")
        batch_files = sorted(reports_dir.glob("batch_results_*.json"), reverse=True)
        if not batch_files:
            print("[ERROR] No batch_results_*.json found in reports/")
            print("        Run: conda run -n deep_sentinel python scripts/batch_run.py <image_dir>")
            sys.exit(1)
        batch_json = str(batch_files[0])
        print(f"[INFO] Auto-selected latest batch: {batch_json}")

    generate_submission(batch_json, threshold)
