# evaluate_frequency_agent.py — 3-field schema, no gan_probability

import sys
sys.path.insert(0, ".")
from agents.frequency_agent import run, validate_output

TEST_IMAGES = {
    "test_fake_stylegan2":  "test_images/test_fake_stylegan2.jpg",
    "test_fake_faceswap":   "test_images/test_fake_faceswap.jpg",
    "test_fake_diffusion":  "test_images/test_fake_diffusion.jpg",
    "test_real_ffhq":       "test_images/test_real_ffhq.jpg",
    "test_compressed_fake": "test_images/test_compressed_fake.jpg",
}

ASSERTIONS = [
    # (field,               image_key,              op,   threshold)
    ("fft_high_anomaly_db", "test_fake_stylegan2",  ">=", 1.0),
    ("anomaly_score",       "test_fake_stylegan2",  ">=", 0.50),
    ("anomaly_score",       "test_real_ffhq",       "<=", 0.35),
    ("anomaly_score",       "test_compressed_fake", ">=", 0.30),
    ("face_cropped",        "test_fake_stylegan2",  "==", True),
    ("face_cropped",        "test_real_ffhq",       "==", True),
]

results = {}
for key, path in TEST_IMAGES.items():
    out = run({"input_type": "image", "path": path})
    results[key] = out
    print(f"\n[{key}]")
    for field, val in out.items():
        print(f"  {field}: {val:.4f}")
    print(f"  validate_output: {validate_output(out)}")

print("\n── ASSERTIONS ──")
passed = 0
for field, img_key, op, thr in ASSERTIONS:
    val    = results[img_key][field]
    ok     = (val >= thr) if op == ">=" else (val <= thr)
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    print(f"  {status}  {field} [{img_key}] {op} {thr}  (actual: {val:.4f})")

print(f"\n{passed}/{len(ASSERTIONS)} assertions passing")
