# L3 Frequency Agent — Paper Reading Order

**Current build target:** `agents/frequency_agent.py` (`STUB_MODE = False`)  
**Goal:** Pass all 7 test assertions in `testing.md` §Agent 03  
**Reference score target:** `anomaly_score ≈ 0.90–0.95` on StyleGAN2 (per DFA-2025-TC-00471 §5.2)

---

## Pipeline State

| Agent | Status |
|---|---|
| Preprocessing (L1) | STUB |
| **Frequency (L3)** | **🔴 ACTIVE BUILD TARGET** |
| Geometry (L2) | STUB |
| Texture (L4) | STUB |
| Biological (L5) | STUB |
| VLM (L6) | STUB |
| Fusion (L7) | STUB |
| Report (L8) | STUB |

**Rule:** Nothing else gets touched until L3 passes all tests in `testing.md`.

---

## Test Assertions You Must Hit

```python
assert result["anomaly_score"]       >= 0.55   # on test_fake_stylegan2.jpg
assert result["fft_anomaly_score"]   >= 0.55   # on test_fake_stylegan2.jpg
assert result["dct_score"]           >= 0.40   # on test_fake_stylegan2.jpg
assert result["phase_anomaly_score"] >= 0.40   # on test_fake_faceswap.jpg
assert result["anomaly_score"]       <= 0.20   # on test_real_ffhq.jpg
assert result["anomaly_score"]       >= 0.30   # on test_compressed_fake.jpg (JPEG q=50)
assert validate_output(result) == True         # all 7 schema fields present, float, [0,1]
```

---

## 🔴 READ FIRST — Implement These Now (one paper per sub-method)

> All code is already synthesized in `frequency-agent-research.md` §4–§7.
> Read the paper to understand the signal, then copy and wire the code block.

### 1. Durall et al. — CVPR 2020
**"Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions"**

- **Sub-method:** Method A — FFT Radial Spectrum
- **Function to write:** `_run_fft_analysis()`
- **Signal:** GAN images violate 1/f spectral roll-off — excess energy at mid/high/ultra frequency bands
- **What to borrow:** Azimuthal radial averaging of 2D FFT magnitude; band energy thresholds
- **Calibration values (from DFA-2025-TC-00471 §5.2):**
  - Mid-band excess: +9.4 dB → anomaly
  - High-band excess: +13.3 dB → anomaly
  - Ultra-high excess: +15.6 dB → StyleGAN2 signature
- **Code:** Already written in `frequency-agent-research.md` §4
- **Maps to assert:** `fft_anomaly_score >= 0.55` on StyleGAN2

---

### 2. Frank et al. — ICML 2020
**"Leveraging Frequency Analysis for Deep Fake Image Recognition"**

- **Sub-method:** Method B — Block DCT Coefficient Distribution
- **Function to write:** `_run_dct_analysis()`
- **Signal:** GAN images show elevated high-frequency AC coefficients in 8×8 block DCT vs real face statistics
- **What to borrow:** Block DCT computation, hf_ratio normalization (0.08 → 0.0, 0.35 → 1.0)
- **GitHub:** https://github.com/RUB-SysSec/GANDCTAnalysis — read DCT normalization logic
- **Code:** Already written in `frequency-agent-research.md` §5
- **Maps to assert:** `dct_score >= 0.40` on StyleGAN2

---

### 3. Tan et al. — AAAI 2024 (FreqNet)
**"Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Learning"**  
arXiv: 2403.07240

- **Sub-method:** Method C — Phase / Amplitude Separation
- **Function to write:** `_run_phase_analysis()`
- **Signal:** Phase spectrum carries blending seam traces that amplitude misses; face-swap operations leave phase gradient discontinuities at boundary regions
- **What to borrow:** Phase gradient computation, wrapping to [−π, π], discontinuity scoring
- **GitHub:** https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection
- **Note:** Borrow the signal extraction logic only — do NOT copy the full CNN architecture
- **Code:** Already written in `frequency-agent-research.md` §6
- **Maps to assert:** `phase_anomaly_score >= 0.40` on FaceSwap

---

### 4. Uddin et al. — Springer Visual Computer 2025 + Wolter et al. — ECML 2022
**Uddin:** "Deepfake Face Detection via Multi-Level Discrete Wavelet Transform and Vision Transformer" (doi: 10.1007/s00371-024-03791-8)  
**Wolter:** "Wavelet-Based Detection of Deepfakes" — frequency-forensics repo

- **Sub-method:** Method D — DWT Subband Energy
- **Function to write:** `_run_dwt_analysis()`
- **Signal:** HH subband energy is naturally low in real faces; GAN upsampling elevates HH energy. LH/HL imbalance indicates directional artifacts from transposed convolution.
- **What to borrow from Uddin:** 2-level Haar DWT + ViT insight — HH/LH/HL subband decomposition
- **What to borrow from Wolter:** Runnable wavelet packet code + pre-computed real/fake coefficient distributions for threshold calibration
- **GitHub (Wolter — use this for actual code):** https://github.com/gan-police/frequency-forensics
- **Note:** Uddin has no public code. Use Wolter's repo for implementation; use Uddin's paper for calibration targets (99.86% on FF++, HH ratio real: 0.02–0.05, fake: 0.08–0.18)
- **Code:** Already written in `frequency-agent-research.md` §7
- **Maps to assert:** `dwt_score` field present, float, in [0,1]

---

## 🟡 READ SECOND — After Baseline Tests Pass

> Do not read these until all 7 test assertions above pass.

### 5. Tan et al. — CVPR 2024 (NPR)
**"Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection"**  
arXiv: 2312.10461

- **Purpose:** Add **Method E** — covers diffusion-generated fakes which current 4 methods miss
- **Signal:** Upsampling operators create measurable neighboring pixel interdependence patterns absent in real photos. Works across 28 generators including GAN + diffusion.
- **Why now:** `test_fake_diffusion.jpg` is in the required test image set. Methods A–D will likely fail on it.
- **What to implement:** NPR feature map extraction as a 5th sub-score in `frequency_agent.py`
- **GitHub:** https://github.com/chuangchuangtan/NPR-DeepfakeDetection
- **Produces:** Neighboring pixel relationship heatmap (interpretable, usable in report)

---

### 6. Wolter et al. — ECML 2022 (frequency-forensics — calibration pass)
**"Wavelet-Based Detection of Deepfakes"**

- **Purpose:** Calibrate `config.json` DWT thresholds using pre-computed real/fake wavelet coefficient distributions
- **Action:** Run their pre-computed analysis on FF++ real/fake splits → extract HH ratio distributions → update `dwt_thresholds` in `config.json`
- **GitHub:** https://github.com/gan-police/frequency-forensics — includes dataset prep + pre-trained models

---

## 🟢 READ THIRD — After L3 Passes All Tests (v2 Upgrade Path)

| # | Paper | Upgrade Target | Notes |
|---|---|---|---|
| 7 | FreqNet full architecture — AAAI 2024 | Method C → HFRI/HFRF learned representation | Replace manual phase gradient with learned CNN feature extractor |
| 8 | Wavelet-CLIP — WACV Workshop 2025 | Method D → CLIP-guided DWT | Best cross-dataset wavelet generalization. GitHub: https://github.com/lalithbharadwajbaru/wavelet-clip |
| 9 | ADD — AAAI 2022 / HiFE — Elsevier 2024 | Methods B+D JPEG robustness | Harden against c40 compressed deepfakes after c23 tests pass |
| 10 | FrePGAN — AAAI 2022 | Training augmentation | Frequency perturbation maps for cross-GAN calibration improvement |

---

## ⚪ SKIP FOR NOW

| Paper | Reason |
|---|---|
| FreqBlender — NeurIPS 2024 | No public code released yet |
| WMamba — arXiv 2025 | No public code released yet |

---

## Alignment Check

| Paper | Aligns to L3? | Maps to Test Assert |
|---|---|---|
| Durall CVPR 2020 | ✅ Direct | `fft_anomaly_score >= 0.55` on StyleGAN2 |
| Frank ICML 2020 | ✅ Direct | `dct_score >= 0.40` on StyleGAN2 |
| FreqNet AAAI 2024 | ✅ Direct | `phase_anomaly_score >= 0.40` on FaceSwap |
| Uddin/Wolter 2025/2022 | ✅ Direct | `dwt_score` in schema, float, [0,1] |
| NPR CVPR 2024 | ✅ Near-term | `test_fake_diffusion.jpg` generalization |
| FrePGAN AAAI 2022 | ✅ Training only | Compressed fake robustness |
| WMamba 2025 | ⚠️ Future | No code available yet |

---

## Immediate Next Action

```
1. Open frequency-agent-research.md §4  →  copy _run_fft_analysis()
2. Open frequency-agent-research.md §5  →  copy _run_dct_analysis()
3. Open frequency-agent-research.md §6  →  copy _run_phase_analysis()
4. Open frequency-agent-research.md §7  →  copy _run_dwt_analysis()
5. Assemble agents/frequency_agent.py with STUB_MODE = False
6. Run all 7 test assertions from testing.md §Agent 03
7. If phase_anomaly_score fails on faceswap → check FreqNet GitHub for phase gradient fix
8. If dwt_score miscalibrated → check Wolter repo for real/fake HH ratio distributions
9. Only after all 7 pass → move to READ SECOND (NPR + calibration)
```
