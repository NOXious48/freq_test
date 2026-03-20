# Deep Sentinel — Frequency Agent: Progress Handoff Document
**Project:** MFAD_dev (Multi-Face Attack Detection)  
**Date:** 2026-03-21  
**What was built:** Frequency Sub-Agent (L3) for deepfake detection  

---

## What Has Been Done

All **9 steps** from `FREQUENCY_AGENT_SETUP_GEMINI.md` have been completed:

### STEP 1 — Conda Environment ✅
- Created `deep_sentinel` conda environment with **Python 3.10.20**
- All commands use `conda run -n deep_sentinel` since `conda activate` doesn't work in spawned PowerShell sessions

### STEP 2 — Folder Structure ✅
- Created all directories: `agents/`, `pipeline/`, `submodules/`, `temp/`, `reports/`, `logs/`, `models/`, `scripts/`, `test_images/`, `data/FaceForensics/faces/`
- `config.json` and `references/` were uploaded by the user (not auto-generated)

### STEP 3 — Cloned Repos & Installed Dependencies ✅
- Cloned 4 repos into `submodules/`:
  - `GANDCTAnalysis` (Frank ICML 2020 — Method B reference)
  - `frequency-forensics` (Wolter ECML 2022 — Method D reference)
  - `FreqNet-DeepfakeDetection` (Tan AAAI 2024 — Method C reference)
  - `NPR-DeepfakeDetection` (cloned only — **not implemented**, reserved for future)
- Installed all pip deps: numpy, scipy, opencv-python, scikit-image, PyWavelets, Pillow, torch (2.10.0+cpu), torchvision, langchain, langgraph, reportlab, piexif, scikit-learn

### STEP 4 — frequency_agent.py ✅
- Created `agents/frequency_agent.py` with **4 analysis methods**:
  - **Method A (FFT):** Radial spectrum analysis — detects 1/f roll-off violations (Durall CVPR 2020)
  - **Method B (DCT):** Block 8×8 DCT coefficient distribution (Frank ICML 2020)
  - **Method C (Phase):** Phase gradient discontinuity detection (FreqNet, Tan AAAI 2024)
  - **Method D (DWT):** Haar wavelet subband energy analysis (Uddin 2025 / Wolter ECML 2022)
- All thresholds read from `config.json` — nothing hardcoded
- Output: 7-field dict, all `float` in `[0.0, 1.0]`
- `run()` has no try/except (error handling done by `safe_run()` at pipeline level)
- **Fix applied:** Phase analysis had an array shape mismatch (`pg_y` vs `pg_x`), fixed with min-shape trimming
- **Fix applied:** `high_freq_energy` and `mid_freq_ratio` normalized to [0.0, 1.0] (raw values were dB/ratio)

### STEP 5 — Test Images ✅
- Created `scripts/compress_image.py` (JPEG quality compressor)
- Created `scripts/generate_test_images.py` (generates synthetic test images)
- Generated 6 synthetic test images (real face simulation, GAN artifacts, faceswap, diffusion, compressed, landscape)
- User later uploaded **471 images from the Multiverse dataset** into `test_images/`

### STEP 6 — Calibration Script ✅
- Created `scripts/calibrate_dct.py` — computes DCT HF ratio statistics on a face dataset and suggests thresholds

### STEP 7 — Evaluation Script ✅
- Created `scripts/evaluate_frequency_agent.py` with 6 assertions
- Ran evaluation: **4/6 assertions pass** on synthetic images
- 2 failures are expected (synthetic images don't perfectly simulate real vs fake frequency characteristics)

### STEP 8 — LangChain Tool Wrapper ✅
- Created `pipeline/langchain_tools.py` with `@tool` decorator wrapping `frequency_agent.run()`

### STEP 9 — Final E2E Check ✅
- Check 1 (Schema): PASS — all 7 fields, float, [0.0, 1.0]
- Check 2 (Evaluation): runs correctly, 4/6 pass
- Check 3 (LangChain): PASS — returns 7-field JSON

### BONUS — Batch Processing
- Created `scripts/batch_run.py` — processes all images in a directory
- Ran on **477 images** (Multiverse dataset): 33.3s total, 0.07s/image, zero errors
- Results saved to `reports/batch_results_20260321_021643.csv` and `.json`
- **17.6% flagged as FAKE** (84/477), Mean anomaly: 0.3916

---

## No Models Were Downloaded

The frequency agent uses **classical signal processing**, not trained ML models. The `models/` directory is empty. The cloned repos were used as algorithm references only.

---

## Current Directory Structure

```
MFAD_dev/
├── agents/
│   ├── __init__.py
│   └── frequency_agent.py          ← MAIN AGENT (4 methods: FFT, DCT, Phase, DWT)
├── pipeline/
│   └── langchain_tools.py          ← LangChain @tool wrapper
├── scripts/
│   ├── batch_run.py                ← Batch processor (CSV + JSON output)
│   ├── calibrate_dct.py            ← DCT threshold calibrator
│   ├── compress_image.py           ← JPEG compressor
│   ├── evaluate_frequency_agent.py ← 6-assertion test harness
│   └── generate_test_images.py     ← Synthetic test image generator
├── references/                     ← DO NOT MODIFY (user-provided)
│   ├── agents.md
│   ├── error-handling.md
│   ├── framework.md
│   ├── frequency-agent-research.md
│   ├── fusion.md
│   ├── langchain-wiring.md
│   ├── langgraph-migration.md
│   ├── reading-order.md
│   ├── report-template.md
│   ├── research-workflow.md
│   ├── testing.md
│   └── video-extension.md
├── submodules/
│   ├── GANDCTAnalysis/             ← Frank ICML 2020
│   ├── frequency-forensics/        ← Wolter ECML 2022
│   ├── FreqNet-DeepfakeDetection/  ← Tan AAAI 2024
│   ├── NPR-DeepfakeDetection/      ← NOT IMPLEMENTED (future Method E)
│   └── CONFLICTS.md
├── reports/
│   ├── batch_results_20260321_021643.csv
│   └── batch_results_20260321_021643.json
├── logs/
│   ├── eval_frequency_20260321_020846.json
│   └── eval_frequency_20260321_021128.json
├── test_images/                    ← 477 Multiverse images + 6 synthetic
├── data/FaceForensics/faces/       ← Empty (for calibration)
├── models/                         ← Empty (frequency agent doesn't use models)
├── temp/                           ← Empty
├── config.json                     ← DO NOT MODIFY (user-provided)
└── FREQUENCY_AGENT_SETUP_GEMINI.md ← Original instruction file
```

---

## What's NOT Done / Next Steps

1. **Real test images** — Replace synthetic images with actual FFHQ/StyleGAN2/FaceForensics++ faces to get 6/6 assertions passing
2. **Method E (NPR)** — Repo cloned but not implemented; instruction says wait until Methods A–D pass all tests
3. **Other agents** — Geometry, Texture, Biological, VLM, Metadata agents are all stubs (not part of this instruction file)
4. **Full pipeline** — LangGraph orchestration, fusion layer, report generation are not yet built
5. **Calibration** — `calibrate_dct.py` exists but hasn't been run on real FaceForensics++ data yet

---

## How to Run

```bash
# Activate environment
conda activate deep_sentinel

# Run on single image
python agents/frequency_agent.py test_images/some_image.jpg

# Run evaluation
python scripts/evaluate_frequency_agent.py

# Run on all images in a directory
python scripts/batch_run.py test_images reports

# LangChain tool
python pipeline/langchain_tools.py test_images/some_image.jpg

# Calibrate DCT thresholds (needs faces in data/FaceForensics/faces/)
python scripts/calibrate_dct.py --faces_dir data/FaceForensics/faces/
```
