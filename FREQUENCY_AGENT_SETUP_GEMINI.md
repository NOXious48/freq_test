# FREQUENCY_AGENT_SETUP_GEMINI.md
# Deep Sentinel — Frequency Sub-Agent (L3)
# Cursor Agent Instruction File — Optimized for Gemini 2.5 Pro
#
# ============================================================
# HOW TO USE:
# 1. Open MFAD_dev/ as workspace root in Cursor
# 2. Switch model to: Gemini 2.5 Pro (High) in Cursor Agent
# 3. Open this file in Cursor
# 4. Type in chat: "Follow FREQUENCY_AGENT_SETUP_GEMINI.md step by step.
#    Start with STEP 1. Do not proceed to next step until verify passes."
# 5. Each step is self-contained — Gemini re-reads context at each step
# ============================================================
#
# ACTIVE BUILD TARGET : agents/frequency_agent.py
# ALL OTHER AGENTS    : DO NOT TOUCH — they are stubs
# MODEL               : Gemini 2.5 Pro (High) — Cursor Agent mode
# ============================================================

---

## GEMINI AGENT PROMPT — PASTE THIS INTO CURSOR CHAT TO START

```
You are working on a deepfake detection project called Deep Sentinel
inside the folder MFAD_dev/.

Your job is to build the Frequency Sub-Agent (L3) by following this
instruction file step by step.

Rules:
- Read the instruction file FREQUENCY_AGENT_SETUP_GEMINI.md fully before starting
- Execute one STEP at a time
- After each step, run the VERIFY block and confirm it passes
- Do not proceed to the next step until verify passes
- Do not modify any file inside references/ or config.json
- Do not implement any agent other than frequency_agent.py

Start with STEP 1 now.
```

---

## PRE-FLIGHT — Read These Files Before Any Step

Gemini: at the start of EACH step that involves writing Python code,
re-read these files. Do not rely on earlier context.

```
READ: MFAD_dev/references/frequency-agent-research.md
      → Algorithm code for all 4 methods is in §4, §5, §6, §7
      → Copy code blocks exactly — do not rewrite

READ: MFAD_dev/config.json
      → All numeric thresholds are under "frequency_agent" key
      → Never hardcode a number — always read from this file

READ: MFAD_dev/references/testing.md
      → Test assertions for L3 are in §Agent 03

READ: MFAD_dev/references/error-handling.md
      → run() must NOT contain try/except
      → safe_run() wrapper handles all exceptions at pipeline level
```

---

## STEP 1 — Create Conda Environment

Gemini: run these terminal commands exactly.

```bash
conda create -n deep_sentinel python=3.10 -y
conda activate deep_sentinel
python --version
```

### VERIFY STEP 1
```bash
python --version
# Must show: Python 3.10.x

conda info --envs
# Must show: deep_sentinel with * active
```

**Confirm:** "STEP 1 complete — Python 3.10 environment created."
**Then proceed to STEP 2.**

---

## STEP 2 — Create Project Folder Structure

Gemini: run these commands from inside MFAD_dev/.

```bash
mkdir -p agents
mkdir -p pipeline
mkdir -p submodules
mkdir -p temp
mkdir -p reports
mkdir -p logs
mkdir -p models
mkdir -p scripts
mkdir -p test_images
mkdir -p data/FaceForensics/faces

touch agents/__init__.py
touch submodules/CONFLICTS.md
```

### VERIFY STEP 2
```bash
ls
# Must show: agents/ pipeline/ submodules/ temp/ reports/
#            logs/ models/ scripts/ test_images/ data/
#            config.json references/
```

**Confirm:** "STEP 2 complete — folder structure created."
**Then proceed to STEP 3.**

---

## STEP 3 — Clone Repos and Install Dependencies

Gemini: clone each repo and install its dependencies immediately after.
If a requirements.txt is missing, install core deps manually.
Log every conflict to submodules/CONFLICTS.md.

### 3a — GANDCTAnalysis (Method B — Frank ICML 2020)

```bash
cd submodules
git clone https://github.com/RUB-SysSec/GANDCTAnalysis
cd GANDCTAnalysis
pip install -r requirements.txt || pip install numpy scipy
cd ../..
echo "GANDCTAnalysis: installed $(date)" >> submodules/CONFLICTS.md
```

### 3b — frequency-forensics (Method D — Wolter ECML 2022)

```bash
cd submodules
git clone https://github.com/gan-police/frequency-forensics
cd frequency-forensics
pip install -r requirements.txt || pip install PyWavelets numpy
cd ../..
echo "frequency-forensics: installed $(date)" >> submodules/CONFLICTS.md
```

### 3c — FreqNet (Method C — Tan AAAI 2024)

```bash
cd submodules
git clone https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection
cd FreqNet-DeepfakeDetection
pip install -r requirements.txt || pip install torch torchvision numpy
cd ../..
echo "FreqNet: installed $(date)" >> submodules/CONFLICTS.md
```

### 3d — NPR (Method E — post-baseline only, clone now do not implement)

```bash
cd submodules
git clone https://github.com/chuangchuangtan/NPR-DeepfakeDetection
cd NPR-DeepfakeDetection
pip install -r requirements.txt || pip install torch torchvision
cd ../..
echo "NPR: cloned only - DO NOT IMPLEMENT until Methods A-D pass tests" >> submodules/CONFLICTS.md
```

### 3e — Core Pipeline Dependencies

```bash
conda activate deep_sentinel

pip install "numpy>=1.24"
pip install "scipy>=1.10"
pip install "opencv-python>=4.8"
pip install "scikit-image>=0.21"
pip install "PyWavelets>=1.4"
pip install "Pillow>=10.0"
pip install "torch>=2.0"
pip install "torchvision>=0.15"
pip install "langchain>=0.1"
pip install "langgraph>=0.0.40"
pip install "reportlab>=4.0"
pip install "piexif>=1.1"
pip install "scikit-learn"
```

### VERIFY STEP 3
```bash
python -c "import numpy, scipy, cv2, pywt, sklearn; print('Core deps OK')"
python -c "import torch; print('PyTorch:', torch.__version__)"
ls submodules/
# Must show: GANDCTAnalysis/ frequency-forensics/
#            FreqNet-DeepfakeDetection/ NPR-DeepfakeDetection/ CONFLICTS.md
```

**Confirm:** "STEP 3 complete — all repos cloned and deps installed."
**Then proceed to STEP 4.**

---

## STEP 4 — Implement agents/frequency_agent.py

Gemini: before writing this file, re-read:
- MFAD_dev/references/frequency-agent-research.md  (§4, §5, §6, §7)
- MFAD_dev/config.json  (key: "frequency_agent")

Create MFAD_dev/agents/frequency_agent.py with this exact content:

```python
# MFAD_dev/agents/frequency_agent.py
# Frequency Sub-Agent (L3) — Deep Sentinel
#
# Methods:
#   A: FFT Radial Spectrum        — Durall et al. CVPR 2020
#   B: Block DCT Coefficients     — Frank et al. ICML 2020
#   C: Phase/Amplitude Separation — FreqNet, Tan et al. AAAI 2024
#   D: DWT Subband Energy         — Uddin 2025 / Wolter ECML 2022
#
# Input : {"input_type": "image", "path": "face_crop_path"}
# Output: 7-field dict, all float in [0.0, 1.0]

import json
import numpy as np
import cv2
from scipy.fftpack import dct as scipy_dct
import pywt
from pathlib import Path

# ── Load config (all thresholds come from here — never hardcode) ──────────────
_CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(_CONFIG_PATH) as f:
    _CFG = json.load(f)["frequency_agent"]

_FFT_CFG   = _CFG["fft_thresholds"]
_DCT_CFG   = _CFG["dct_thresholds"]
_PHASE_CFG = _CFG["phase_thresholds"]
_DWT_CFG   = _CFG["dwt_thresholds"]
_WEIGHTS   = _CFG["submethod_weights"]
_IMG_SIZE  = _CFG["image_size"]
_FFT_BANDS = _CFG["fft_bands"]

# ── Agent constants ───────────────────────────────────────────────────────────
STUB_MODE = False

STUB_OUTPUT = {
    "fft_anomaly_score":   0.0,
    "high_freq_energy":    0.0,
    "mid_freq_ratio":      0.0,
    "dct_score":           0.0,
    "phase_anomaly_score": 0.0,
    "dwt_score":           0.0,
    "anomaly_score":       0.0,
}

SCHEMA = STUB_OUTPUT.keys()


def validate_output(output: dict) -> bool:
    """All 7 fields present, float type, in [0.0, 1.0]."""
    if not all(k in output for k in SCHEMA):
        return False
    for k in SCHEMA:
        v = output[k]
        if not isinstance(v, float):
            return False
        if not (0.0 <= v <= 1.0):
            return False
    return True


def run(input: dict) -> dict:
    """
    Frequency Agent entrypoint — max 30 lines.
    Orchestrates 4 sub-methods and returns weighted anomaly_score.
    Does NOT catch exceptions — safe_run() handles that at pipeline level.

    Args:
        input: {"input_type": "image", "path": str}
    Returns:
        dict with 7 fields, all float in [0.0, 1.0]
    """
    if STUB_MODE:
        return STUB_OUTPUT.copy()

    image_path = input["path"]

    fft_result   = _run_fft_analysis(image_path)
    dct_result   = _run_dct_analysis(image_path)
    phase_result = _run_phase_analysis(image_path)
    dwt_result   = _run_dwt_analysis(image_path)

    anomaly_score = (
        _WEIGHTS["fft"]   * fft_result["fft_anomaly_score"]   +
        _WEIGHTS["dct"]   * dct_result["dct_score"]           +
        _WEIGHTS["phase"] * phase_result["phase_anomaly_score"] +
        _WEIGHTS["dwt"]   * dwt_result["dwt_score"]
    )

    output = {
        "fft_anomaly_score":   fft_result["fft_anomaly_score"],
        "high_freq_energy":    fft_result["high_freq_energy"],
        "mid_freq_ratio":      fft_result["mid_freq_ratio"],
        "dct_score":           dct_result["dct_score"],
        "phase_anomaly_score": phase_result["phase_anomaly_score"],
        "dwt_score":           dwt_result["dwt_score"],
        "anomaly_score":       float(np.clip(anomaly_score, 0.0, 1.0)),
    }

    assert validate_output(output), f"Schema validation failed: {output}"
    return output


# ── Method A: FFT Radial Spectrum (Durall et al. CVPR 2020) ──────────────────

def _run_fft_analysis(image_path: str) -> dict:
    """
    Detects violation of 1/f spectral roll-off.
    GAN images show excess energy at mid/high/ultra frequency bands.
    Reference: Durall et al. CVPR 2020
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32)

    f       = np.fft.fft2(img)
    fshift  = np.fft.fftshift(f)
    log_mag = 20 * np.log10(np.abs(fshift) + 1e-8)

    h, w   = log_mag.shape
    cy, cx = h // 2, w // 2
    Y, X   = np.ogrid[:h, :w]
    R      = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    max_r  = min(cy, cx)

    radial_mean = np.array([
        log_mag[R == r].mean() if (R == r).any() else 0.0
        for r in range(max_r)
    ])

    low_end  = int(_FFT_BANDS["low_pct"]  * max_r)
    mid_end  = int(_FFT_BANDS["mid_pct"]  * max_r)
    high_end = int(_FFT_BANDS["high_pct"] * max_r)

    low_e   = radial_mean[:low_end].mean()
    mid_e   = radial_mean[low_end:mid_end].mean()
    high_e  = radial_mean[mid_end:high_end].mean()
    ultra_e = radial_mean[high_end:].mean()

    mid_excess   = max(0.0, mid_e   - (low_e + _FFT_CFG["mid_expected_offset"]))
    high_excess  = max(0.0, high_e  - (low_e + _FFT_CFG["high_expected_offset"]))
    ultra_excess = max(0.0, ultra_e - (low_e + _FFT_CFG["ultra_expected_offset"]))

    norm    = _FFT_CFG["normalization_factor"]
    fft_raw = (0.3 * mid_excess + 0.4 * high_excess + 0.3 * ultra_excess) / norm

    return {
        "fft_anomaly_score": float(np.clip(fft_raw, 0.0, 1.0)),
        "high_freq_energy":  float(high_e),
        "mid_freq_ratio":    float(mid_e / (low_e + 1e-8)),
    }


# ── Method B: Block DCT Coefficient Distribution (Frank et al. ICML 2020) ────

def _run_dct_analysis(image_path: str) -> dict:
    """
    Detects abnormal AC coefficient distribution in 8x8 DCT blocks.
    Real HF ratio ~0.08-0.12. GAN HF ratio ~0.18-0.35.
    Reference: Frank et al. ICML 2020
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32)

    bs        = _DCT_CFG["block_size"]
    h, w      = img.shape
    dct_coeffs = []

    for i in range(0, h - bs + 1, bs):
        for j in range(0, w - bs + 1, bs):
            block     = img[i:i+bs, j:j+bs]
            block_dct = scipy_dct(scipy_dct(block.T, norm='ortho').T, norm='ortho')
            dct_coeffs.append(block_dct.flatten())

    coeff_means = np.array(dct_coeffs).mean(axis=0)
    hf_energy   = np.abs(coeff_means[32:]).mean()
    lf_energy   = np.abs(coeff_means[1:16]).mean()
    hf_ratio    = hf_energy / (lf_energy + 1e-8)

    lo = _DCT_CFG["real_hf_ratio_min"]
    hi = _DCT_CFG["fake_hf_ratio_max"]

    return {
        "dct_score":      float(np.clip((hf_ratio - lo) / (hi - lo), 0.0, 1.0)),
        "hf_coeff_ratio": float(hf_ratio),
    }


# ── Method C: Phase/Amplitude Separation (FreqNet, Tan et al. AAAI 2024) ─────

def _run_phase_analysis(image_path: str) -> dict:
    """
    Detects phase gradient discontinuities at face-swap blending boundaries.
    Real discontinuity ~0.8-1.2. Fakes ~1.8-3.5.
    Reference: FreqNet, Tan et al. AAAI 2024
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32)

    fshift    = np.fft.fftshift(np.fft.fft2(img))
    amplitude = np.abs(fshift)
    phase     = np.angle(fshift)

    pg_y = np.arctan2(np.sin(np.diff(phase, axis=0)), np.cos(np.diff(phase, axis=0)))
    pg_x = np.arctan2(np.sin(np.diff(phase, axis=1)), np.cos(np.diff(phase, axis=1)))

    phase_disc = np.sqrt(pg_y[:-1, :]**2 + pg_x[:, :-1]**2).mean()

    h, w   = amplitude.shape
    cy, cx = h // 2, w // 2
    r      = _PHASE_CFG["center_mask_radius_px"]
    mask   = np.zeros_like(amplitude, dtype=bool)
    mask[cy-r:cy+r, cx-r:cx+r] = True
    amp_hf_ratio = amplitude[~mask].mean() / (amplitude[mask].mean() + 1e-8)

    lo = _PHASE_CFG["real_discontinuity_min"]
    hi = _PHASE_CFG["fake_discontinuity_max"]

    return {
        "phase_anomaly_score": float(np.clip((phase_disc - lo) / (hi - lo), 0.0, 1.0)),
        "phase_discontinuity": float(phase_disc),
        "amplitude_hf_ratio":  float(amp_hf_ratio),
    }


# ── Method D: DWT Subband Energy (Uddin 2025 / Wolter ECML 2022) ─────────────

def _run_dwt_analysis(image_path: str) -> dict:
    """
    Detects elevated HH subband energy from GAN transposed convolution artifacts.
    Real HH ratio ~0.02-0.05. GAN HH ratio ~0.08-0.18.
    Reference: Uddin et al. 2025, Wolter et al. ECML 2022
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32) / 255.0

    wavelet          = _DWT_CFG["wavelet"]
    LL,  (LH,  HL,  HH)  = pywt.dwt2(img, wavelet)
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL,  wavelet)

    def energy(x): return float(np.mean(x**2))

    e_LL, e_LH, e_HL, e_HH = energy(LL), energy(LH), energy(HL), energy(HH)
    e_HH2 = energy(HH2)

    total    = e_LL + e_LH + e_HL + e_HH + 1e-8
    hh_ratio = (e_HH + e_HH2) / total
    imbalance = abs(e_LH - e_HL) / (e_LH + e_HL + 1e-8)

    lo = _DWT_CFG["real_hh_ratio_min"]
    hi = _DWT_CFG["fake_hh_ratio_max"]

    return {
        "dwt_score":        float(np.clip(0.6*(hh_ratio-lo)/(hi-lo) + 0.4*imbalance, 0.0, 1.0)),
        "hh_ratio":         float(hh_ratio),
        "lh_hl_imbalance":  float(imbalance),
        "subband_energies": {"LL": e_LL, "LH": e_LH, "HL": e_HL, "HH": e_HH},
    }


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agents/frequency_agent.py <image_path>")
        sys.exit(1)
    result = run({"input_type": "image", "path": sys.argv[1]})
    print(json.dumps(result, indent=2))
    print("validate_output:", validate_output(result))
```

### VERIFY STEP 4
```bash
python agents/frequency_agent.py test_images/test_real_ffhq.jpg
# Must return: JSON with exactly 7 fields
# Must print:  validate_output: True
# All values must be float in [0.0, 1.0]
```

**Confirm:** "STEP 4 complete — frequency_agent.py created and validated."
**Then proceed to STEP 5.**

---

## STEP 5 — Prepare Test Images

Gemini: create the compress helper script, then guide the user to
place the 6 required images in test_images/.

```python
# Create MFAD_dev/scripts/compress_image.py
import cv2, sys

def compress(input_path: str, quality: int = 50):
    img         = cv2.imread(input_path)
    output_path = input_path.replace(".jpg", f"_q{quality}.jpg")
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    compress(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50)
```

Required images — place in MFAD_dev/test_images/:

| Filename | Where to get it |
|---|---|
| test_real_ffhq.jpg | https://github.com/NVlabs/ffhq-dataset — any image from images1024x1024/ |
| test_fake_stylegan2.jpg | https://thispersondoesnotexist.com — save any face |
| test_fake_faceswap.jpg | FaceForensics++ c23 — https://github.com/ondyari/FaceForensics |
| test_fake_diffusion.jpg | https://huggingface.co/spaces/stabilityai/stable-diffusion |
| test_compressed_fake.jpg | Run: python scripts/compress_image.py test_images/test_fake_stylegan2.jpg 50 |
| test_no_face.jpg | Any landscape/object photo |

```bash
# Generate compressed fake after test_fake_stylegan2.jpg is in place
python scripts/compress_image.py test_images/test_fake_stylegan2.jpg 50
mv test_images/test_fake_stylegan2_q50.jpg test_images/test_compressed_fake.jpg
```

### VERIFY STEP 5
```bash
ls test_images/
# Must show all 6 files
```

**Confirm:** "STEP 5 complete — test images in place."
**Then proceed to STEP 6.**

---

## STEP 6 — Calibration Script

Gemini: create MFAD_dev/scripts/calibrate_dct.py

```python
# MFAD_dev/scripts/calibrate_dct.py
import argparse, json, numpy as np, cv2
from pathlib import Path
from scipy.fftpack import dct as scipy_dct

def compute_hf_ratio(path, block_size=8):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    h, w = img.shape
    coeffs = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            b = img[i:i+block_size, j:j+block_size]
            coeffs.append(scipy_dct(scipy_dct(b.T, norm='ortho').T, norm='ortho').flatten())
    means = np.array(coeffs).mean(axis=0)
    return float(np.abs(means[32:]).mean() / (np.abs(means[1:16]).mean() + 1e-8))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces_dir", default="data/FaceForensics/faces/")
    parser.add_argument("--max_images", type=int, default=2000)
    parser.add_argument("--config_out", default="config_calibrated.json")
    args = parser.parse_args()

    images = list(Path(args.faces_dir).glob("*.jpg"))[:args.max_images]
    if not images:
        print(f"[ERROR] No images in {args.faces_dir}"); return

    ratios = [r for img in images if (r := compute_hf_ratio(img)) is not None]
    ratios = np.array(ratios)
    mean, std = ratios.mean(), ratios.std()

    print(f"Mean HF ratio: {mean:.4f}  Std: {std:.4f}")
    print(f"Suggested real_max: {mean + 2*std:.4f}")
    print(f"Suggested fake_min: {mean + 4*std:.4f}")

    try:
        config = json.load(open("config.json"))
    except FileNotFoundError:
        config = {}

    config.setdefault("frequency_agent", {})["dct_reference"] = {
        "mean_hf_ratio": float(mean), "std_hf_ratio": float(std),
        "real_hf_ratio_max": float(mean + 2*std),
        "fake_hf_ratio_min": float(mean + 4*std),
        "n_images": len(ratios),
    }
    json.dump(config, open(args.config_out, "w"), indent=2)
    print(f"Saved to {args.config_out}")

if __name__ == "__main__":
    main()
```

### VERIFY STEP 6
```bash
python scripts/calibrate_dct.py --help
# Must show usage message with no errors
```

**Confirm:** "STEP 6 complete — calibrate_dct.py created."
**Then proceed to STEP 7.**

---

## STEP 7 — Evaluation Script

Gemini: create MFAD_dev/scripts/evaluate_frequency_agent.py

```python
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
        status = "PASS ✓" if ok else "FAIL ✗"
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
```

### VERIFY STEP 7
```bash
python scripts/evaluate_frequency_agent.py
# Must: print scores per image, show PASS/FAIL per assertion, save log
# Target: 7/7 PASS
```

**Confirm:** "STEP 7 complete — evaluation script created and run."
**Then proceed to STEP 8.**

---

## STEP 8 — LangChain Tool Wrapper

Gemini: create MFAD_dev/pipeline/langchain_tools.py

```python
# MFAD_dev/pipeline/langchain_tools.py
# Only frequency_tool is active — all other agents are stubs
import json
from langchain.tools import tool

@tool
def frequency_tool(face_crop_path: str) -> str:
    """
    Analyze a face crop for GAN frequency artifacts.
    Runs FFT radial spectrum, Block DCT, Phase analysis, DWT subband energy.
    Returns JSON string with 7 fields, anomaly_score in [0.0, 1.0].
    """
    from agents.frequency_agent import run
    return json.dumps(run({"input_type": "image", "path": face_crop_path}))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline/langchain_tools.py <image_path>")
        sys.exit(1)
    print(frequency_tool.invoke(sys.argv[1]))
```

### VERIFY STEP 8
```bash
python pipeline/langchain_tools.py test_images/test_real_ffhq.jpg
# Must return: JSON string with 7 fields
```

**Confirm:** "STEP 8 complete — LangChain tool wired."
**Then proceed to STEP 9.**

---

## STEP 9 — Final End-to-End Check

Gemini: run all 3 checks. All must pass.

```bash
# Check 1 — Schema
python -c "
from agents.frequency_agent import run, validate_output
r = run({'input_type': 'image', 'path': 'test_images/test_real_ffhq.jpg'})
assert validate_output(r), 'Schema FAIL'
assert all(isinstance(v, float) for v in r.values()), 'Type FAIL'
assert all(0.0 <= v <= 1.0 for v in r.values()), 'Range FAIL'
print('Schema check: PASS')
"

# Check 2 — Full evaluation
python scripts/evaluate_frequency_agent.py

# Check 3 — LangChain smoke test
python pipeline/langchain_tools.py test_images/test_fake_stylegan2.jpg
```

**If any check fails:**
- Schema fail → re-read frequency-agent-research.md §8 and fix output dict
- Score too low → re-read config.json thresholds and verify they loaded correctly
- Import error → check pip installs from STEP 3

---

## TARGET SCORES

| Image | Field | Must Be |
|---|---|---|
| test_fake_stylegan2 | anomaly_score | >= 0.55 |
| test_fake_stylegan2 | fft_anomaly_score | >= 0.55 |
| test_fake_stylegan2 | dct_score | >= 0.40 |
| test_fake_faceswap | phase_anomaly_score | >= 0.40 |
| test_real_ffhq | anomaly_score | <= 0.20 |
| test_compressed_fake | anomaly_score | >= 0.30 |
| any image | validate_output | True |

Fully calibrated target: anomaly_score ≈ 0.90–0.95 on StyleGAN2

---

## CONSTRAINTS — GEMINI MUST FOLLOW

- Do NOT modify references/ or config.json
- Do NOT implement any agent other than frequency_agent.py
- Do NOT add try/except to run() — error handling is in safe_run()
- Do NOT hardcode any number — all thresholds from config.json
- Do NOT implement Method E (NPR) until all 7 assertions pass
- run() must stay under 30 lines
- All output values must be float in [0.0, 1.0]
- Generate one file per step — confirm verify passes before next step
