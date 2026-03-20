# MFAD_dev/scripts/evaluate_frequency_agent.py
import json, sys, os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.frequency_agent import run, validate_output

TEST_IMAGES = {
    "test_fake_stylegan2":  {"path": "test_images/test_fake_stylegan2.jpg",  "label": 1},
    "test_fake_faceswap":   {"path": "test_images/test_fake_faceswap.jpg",   "label": 1},
    "test_fake_diffusion":  {"path": "test_images/test_fake_diffusion.jpg",  "label": 1},
    "test_real_ffhq":       {"path": "test_images/test_real_ffhq.jpg",       "label": 0},
    "test_compressed_fake": {"path": "test_images/test_compressed_fake.jpg", "label": 1},
}

ASSERTIONS = [
    ("anomaly_score",       "test_fake_stylegan2",  ">=", 0.55),
    ("fft_anomaly_score",   "test_fake_stylegan2",  ">=", 0.55),
    ("dct_score",           "test_fake_stylegan2",  ">=", 0.40),
    ("phase_anomaly_score", "test_fake_faceswap",   ">=", 0.40),
    ("anomaly_score",       "test_real_ffhq",       "<=", 0.20),
    ("anomaly_score",       "test_compressed_fake", ">=", 0.30),
]

def main():
    print("=" * 55)
    print("Frequency Agent Evaluation — Deep Sentinel L3")
    print("=" * 55)

    results = {}
    passed  = 0
    labels, scores = [], []

    for name, meta in TEST_IMAGES.items():
        if not Path(meta["path"]).exists():
            print(f"\n[SKIP] {name} — not found"); continue
        print(f"\n[Running] {name}")
        result = run({"input_type": "image", "path": meta["path"]})
        results[name] = result
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k:<25} {v:.4f}")
        print(f"  validate_output: {validate_output(result)}")
        labels.append(meta["label"])
        scores.append(result.get("anomaly_score", 0.5))

    print("\n" + "=" * 55)
    print("ASSERTIONS")
    print("=" * 55)
    for field, img, op, thr in ASSERTIONS:
        if img not in results or "error" in results.get(img, {}):
            print(f"  [SKIP] {field} on {img}"); continue
        val = results[img].get(field, 0.0)
        ok  = (val >= thr) if op == ">=" else (val <= thr)
        if ok: passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {field} {op} {thr}  [{img}]  actual={val:.4f}")

    print(f"\nResult: {passed}/{len(ASSERTIONS)} passed")

    if len(set(labels)) == 2:
        from sklearn.metrics import roc_auc_score
        print(f"AUC-ROC: {roc_auc_score(labels, scores):.4f}")

    os.makedirs("logs", exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"logs/eval_frequency_{ts}.json"
    json.dump({"timestamp": ts, "passed": passed,
               "total": len(ASSERTIONS), "results": results},
              open(out, "w"), indent=2, default=str)
    print(f"Log saved: {out}")

if __name__ == "__main__":
    main()
