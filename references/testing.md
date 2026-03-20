# Testing Specification — Deep Sentinel Pipeline

**Purpose:** Defines pass criteria, test images, and validation procedures for each agent.
An agent is NOT considered complete until all tests in its section pass.

---

## Global Rules

Every agent must satisfy these conditions before being marked non-stub:

1. `validate_output(result)` returns `True` — all schema fields present, no missing keys
2. All score fields are `float` type and within `[0.0, 1.0]`
3. On a known **fake** image: `anomaly_score >= 0.50`
4. On a known **real** image: `anomaly_score <= 0.25`
5. No exceptions raised on a JPEG-compressed input (quality=50)
6. Execution time under 30 seconds per image (excluding GPU model load)

---

## Test Image Set

### Required Images (obtain before testing any agent)

| ID | Source | Type | Use |
|---|---|---|---|
| `test_fake_stylegan2.jpg` | FFHQ fake set / FF++ | StyleGAN2 generated face | Primary fake test |
| `test_fake_faceswap.jpg` | FaceForensics++ c23 | FaceSwap face-swap | Secondary fake test |
| `test_fake_diffusion.jpg` | Stable Diffusion / DALL-E 3 | Diffusion-generated face | Generalization test |
| `test_real_ffhq.jpg` | FFHQ (verified authentic) | Real photograph | Primary real test |
| `test_real_celeb.jpg` | CelebA-HQ (verified authentic) | Real photograph | Secondary real test |
| `test_compressed_fake.jpg` | Any fake, re-saved JPEG quality=50 | Compressed fake | Compression robustness |
| `test_no_face.jpg` | Any image without a face | No face | Error handling test |

### Where to Obtain Test Images

- FaceForensics++ dataset: https://github.com/ondyari/FaceForensics
- FFHQ dataset: https://github.com/NVlabs/ffhq-dataset
- CelebA-HQ: https://github.com/tkarras/progressive_growing_of_gans

---

## Agent 01 — Preprocessing

### Pass Criteria

```python
# Face detection test
result = run({"input_type": "image", "path": "test_real_ffhq.jpg"})
assert result["face_detected"] == True
assert result["face_crop_path"] is not None
assert result["normalized_path"] is not None
assert result["landmarks_path"] is not None
assert result["original_path"] is not None     # critical — must be present
assert len(result["bbox"]) == 4

# No-face test — must fail gracefully
result = run({"input_type": "image", "path": "test_no_face.jpg"})
assert result["face_detected"] == False
assert "error" in result or result["anomaly_score"] == 1.0

# Schema validation
assert validate_output(result)
```

### What to Verify Manually

Open `face_crop_path` and `normalized_path` images and confirm they are correctly cropped
and resized to 224×224. Open `landmarks_path` JSON and confirm 68 landmark points are present.

---

## Agent 03 — Frequency (CURRENT BUILD TARGET)

### Pass Criteria

```python
# Fake image — must detect
result = run({"input_type": "image", "path": "test_fake_stylegan2.jpg"})
assert result["anomaly_score"] >= 0.55,        "StyleGAN2 must score > 0.55"
assert result["fft_anomaly_score"] >= 0.55,    "FFT must catch StyleGAN2 spectral excess"
assert result["dct_score"] >= 0.40,            "DCT must flag abnormal AC coefficients"

# Real image — must not false-positive
result = run({"input_type": "image", "path": "test_real_ffhq.jpg"})
assert result["anomaly_score"] <= 0.20,        "Real face must score < 0.20"
assert result["fft_anomaly_score"] <= 0.15,    "FFT must be low on real face"

# FaceSwap fake test
result = run({"input_type": "image", "path": "test_fake_faceswap.jpg"})
assert result["phase_anomaly_score"] >= 0.40,  "Phase must catch face-swap seams"

# Compression robustness
result = run({"input_type": "image", "path": "test_compressed_fake.jpg"})
assert result["anomaly_score"] >= 0.30,        "Must still detect on JPEG q=50"

# Schema and type validation
assert validate_output(result)
expected_keys = ["fft_anomaly_score", "high_freq_energy", "mid_freq_ratio",
                 "dct_score", "phase_anomaly_score", "dwt_score", "anomaly_score"]
for key in expected_keys:
    assert key in result,               f"Missing field: {key}"
    assert isinstance(result[key], float), f"{key} must be float"
    assert 0.0 <= result[key] <= 1.0,   f"{key} out of range [0,1]"
```

### Expected Score Ranges

These ranges come from the calibration strategy in `frequency-agent-research.md`:

| Score Field | Real Face | Fake Face |
|---|---|---|
| `fft_anomaly_score` | 0.00 – 0.15 | 0.55 – 0.95 |
| `dct_score` | 0.00 – 0.12 | 0.45 – 0.90 |
| `phase_anomaly_score` | 0.00 – 0.20 | 0.40 – 0.85 |
| `dwt_score` | 0.00 – 0.10 | 0.35 – 0.80 |
| `anomaly_score` | 0.00 – 0.15 | 0.50 – 0.95 |

### Reference Target

The reference report DFA-2025-TC-00471 shows Frequency-Domain Spectral Anomaly at **91.2%**.
A fully calibrated agent on a StyleGAN2 fake should produce `anomaly_score` ≈ 0.90–0.95.

---

## Agent 02 — Geometry

### Pass Criteria

```python
result = run({"input_type": "image", "path": "test_fake_stylegan2.jpg"})
assert result["anomaly_score"] >= 0.50
assert result["symmetry_score"] >= 0.50       # GAN faces are asymmetric

result = run({"input_type": "image", "path": "test_real_ffhq.jpg"})
assert result["anomaly_score"] <= 0.25

assert validate_output(result)
for key in ["symmetry_score", "landmark_anomaly_score", "eye_distance_ratio",
            "jaw_irregularity", "anomaly_score"]:
    assert 0.0 <= result[key] <= 1.0
```

### Reference Target

Reference report §5.1: symmetry index 0.74 (−19.6%), jaw curvature 11.2°.
Geometry score in reference: **88.4%**.

---

## Agent 04 — Texture

### Pass Criteria

```python
result = run({"input_type": "image", "path": "test_fake_stylegan2.jpg"})
assert result["anomaly_score"] >= 0.50
assert result["emd_score"] >= 0.40            # neck-face boundary EMD should be high

result = run({"input_type": "image", "path": "test_real_ffhq.jpg"})
assert result["anomaly_score"] <= 0.25

assert validate_output(result)
```

### Reference Target

Reference report §5.3: neck-face EMD 0.274 (3.4σ above mean).
Texture score in reference: **89.5%**.

---

## Agent 05 — Biological

### Pass Criteria

```python
result = run({"input_type": "image", "path": "test_fake_stylegan2.jpg"})
assert result["anomaly_score"] >= 0.45        # lower bar — this is the hardest agent
assert result["corneal_reflection_score"] >= 0.40

result = run({"input_type": "image", "path": "test_real_ffhq.jpg"})
assert result["anomaly_score"] <= 0.30        # slightly relaxed bar

assert validate_output(result)
```

### Reference Target

Reference report §5.5: rPPG SNR 0.09 (authentic >0.45), corneal deviation 14.3°.
Biological score in reference: **82.6%** (lowest single score — expected).

---

## Agent 06 — VLM Explainability

### Pass Criteria

```python
result = run({"input_type": "image", "path": "test_fake_stylegan2.jpg"})
assert result["vlm_verdict"] in ["FAKE", "UNCERTAIN"]
assert result["anomaly_score"] >= 0.50
assert result["gradcam_path"] is not None
assert len(result["caption"]) > 20            # must produce a real caption

result = run({"input_type": "image", "path": "test_real_ffhq.jpg"})
assert result["vlm_verdict"] in ["REAL", "UNCERTAIN"]
assert result["anomaly_score"] <= 0.30

assert validate_output(result)
```

### Reference Target

Reference report §5.4: RED zone salience 0.91, GAN probability 0.93.
VLM score in reference: **93.1%**.

---

## Agent 07 (new) — Metadata Forensics

### Pass Criteria

```python
# GAN-generated image has no EXIF camera data
result = run({"input_type": "image", "path": "test_fake_stylegan2.jpg"})
assert result["anomaly_score"] >= 0.70        # GAN images have no EXIF — high score
assert result["exif_score"] >= 0.80           # missing camera data is strong signal

result = run({"input_type": "image", "path": "test_real_ffhq.jpg"})
assert result["anomaly_score"] <= 0.30

assert validate_output(result)
```

### Reference Target

Reference report §5.6: missing EXIF, non-standard JPEG quantisation, ELA chi²=847.
Metadata score in reference: **97.3%** (highest single score).

---

## Agent 08 — Fusion

### Pass Criteria

```python
# Simulate all-fake agent scores
fake_scores = {
    "geometry": 0.884, "frequency": 0.912,
    "texture": 0.895,  "biological": 0.826,
    "vlm": 0.931,      "metadata": 0.973
}
result = run(fake_scores)
assert result["verdict"] == "FAKE"
assert result["final_score"] >= 0.65

# Simulate all-real agent scores
real_scores = {
    "geometry": 0.05, "frequency": 0.08,
    "texture": 0.06,  "biological": 0.10,
    "vlm": 0.07,      "metadata": 0.05
}
result = run(real_scores)
assert result["verdict"] == "REAL"
assert result["final_score"] <= 0.35

# Test with one failed agent (error score = 0.5)
mixed_scores = {**fake_scores, "biological": 0.5}  # biological failed
result = run(mixed_scores)
assert result["verdict"] == "FAKE"   # should still detect despite one neutral score
```

### Reference Target

Reference report §6: Final Score 95.0%, 95% CI [93.1%–96.6%], ECE 0.014.

---

## Agent 09 — Report

### Pass Criteria

1. PDF file is created at the `report_path` location
2. PDF is openable and not corrupted (check file size > 50KB)
3. PDF contains all 10 required sections (verify by page count ≥ 8)
4. `report_id` matches format `DFA-{YYYY}-TC-{6_char_hex}`
5. `generated_at` is valid ISO8601 timestamp

---

## Running the Full Pipeline (End-to-End Test)

```python
from pipeline.runner import run_pipeline

result = run_pipeline("test_fake_stylegan2.jpg")

# Must produce master output schema
required_keys = ["preprocessing", "geometry", "frequency", "texture",
                 "biological", "vlm", "metadata", "fusion", "report_path"]
for key in required_keys:
    assert key in result, f"Missing top-level key: {key}"

# Fusion must produce a verdict
assert result["fusion"]["verdict"] in ["FAKE", "REAL", "UNCERTAIN"]
assert 0.0 <= result["fusion"]["final_score"] <= 1.0

print("Full pipeline test PASSED")
print(f"Verdict: {result['fusion']['verdict']}")
print(f"Score:   {result['fusion']['final_score']:.3f}")
print(f"Report:  {result['report_path']}")
```
