# Agent Implementation Reference

## Table of Contents
1. [Preprocessing Agent](#1-preprocessing-agent)
2. [Geometry Agent](#2-geometry-agent)
3. [Frequency Agent](#3-frequency-agent)
4. [Texture Agent](#4-texture-agent)
5. [Biological Agent](#5-biological-agent)
6. [VLM Explainability Agent](#6-vlm-explainability-agent)
7. [Fusion Agent](#7-fusion-agent)
8. [Report Agent](#8-report-agent)

---

## 1. Preprocessing Agent

**Purpose:** Detect and crop face, extract landmarks, normalize for downstream agents.
All other agents depend on this output — run it first and fail fast if face detection fails.

**Output Schema:**
```json
{
  "face_detected": true,
  "face_crop_path": "string",
  "normalized_path": "string",
  "landmarks_path": "string",
  "bbox": [0, 0, 0, 0],
  "anomaly_score": 0.0
}
```

**Implementation Notes:**
- Use MediaPipe FaceMesh for landmark extraction (468 landmarks)
- Use dlib `get_frontal_face_detector` as fallback if MediaPipe fails
- Normalize face to 224x224 for EfficientNet/XceptionNet compatibility
- Save crops to a temp directory; pass paths (not arrays) between agents
- `anomaly_score` here reflects detection confidence inversion (low confidence = higher score)

**Key Libraries:** `mediapipe`, `dlib`, `opencv-python`

---

## 2. Geometry Agent

**Purpose:** Detect facial asymmetry and landmark irregularities characteristic of GAN faces.

**Output Schema:**
```json
{
  "symmetry_score": 0.0,
  "landmark_anomaly_score": 0.0,
  "eye_distance_ratio": 0.0,
  "jaw_irregularity": 0.0,
  "anomaly_score": 0.0
}
```

**Implementation Notes:**
- Use dlib 68-point landmarks from preprocessing output
- Symmetry: mirror left/right landmark distances, compute mean absolute deviation
- Landmark anomaly: compare to a statistical model of real face proportions
- Eye distance ratio: interpupillary distance / face width (GAN faces often distort this)
- `anomaly_score` = weighted combination of sub-scores, range [0, 1]

**Research Basis:** 
- "Exposing DeepFake Videos By Detecting Face Warping Artifacts" (Li et al., 2018)

**Key Libraries:** `dlib`, `numpy`, `scipy`

---

## 3. Frequency Agent

**Purpose:** Detect high-frequency GAN artifacts and unnatural spectral patterns invisible to the human eye.

**Output Schema:**
```json
{
  "fft_anomaly_score": 0.0,
  "high_freq_energy": 0.0,
  "mid_freq_ratio": 0.0,
  "dct_score": 0.0,
  "anomaly_score": 0.0
}
```

**Implementation Notes:**
- Apply 2D FFT to grayscale face crop
- Compute radial frequency spectrum; GAN images often show grid-like artifacts at specific frequencies
- High-freq energy: sum of magnitudes in outer 30% of frequency spectrum
- Mid-freq ratio: mid-band energy / total energy
- DCT: compute block DCT (8x8), compare coefficient distribution to natural image statistics
- Normalize all scores to [0, 1] using empirical thresholds from FaceForensics++

**Research Basis:**
- "Watching the Big Brother: Detecting Audio-Visual Deepfakes" 
- "Unmasking DeepFakes with simple Features" (Durall et al., 2020)

**Key Libraries:** `numpy` (fft2, fftshift), `scipy.fftpack`

---

## 4. Texture Agent

**Purpose:** Detect unnatural skin texture patterns — GAN-generated faces often have hyper-smooth or inconsistent texture.

**Output Schema:**
```json
{
  "lbp_score": 0.0,
  "gabor_score": 0.0,
  "emd_score": 0.0,
  "smoothness_index": 0.0,
  "anomaly_score": 0.0
}
```

**Implementation Notes:**
- **LBP (Local Binary Patterns):** Extract LBP histogram from face region. Compare to reference distribution of real faces using chi-squared distance.
- **Gabor Filters:** Apply bank of Gabor filters at 4 orientations × 3 scales. Compute energy map. GAN faces show abnormal orientation consistency.
- **EMD (Earth Mover's Distance):** Compare texture histogram to reference real-face distribution.
- **Smoothness Index:** Variance of Laplacian on skin patches — real faces have natural texture variance; GAN faces are often unnaturally smooth in non-feature regions.

**Research Basis:**
- "FaceForensics++: Learning to Detect Manipulated Facial Images" (Rossler et al., 2019)

**Key Libraries:** `scikit-image` (local_binary_pattern, gabor), `scipy.stats` (wasserstein_distance), `opencv-python`

---

## 5. Biological Agent

**Purpose:** Detect absence of physiological signals present in real faces — heartbeat pulse, natural corneal reflections.

**Output Schema:**
```json
{
  "rppg_score": 0.0,
  "corneal_reflection_score": 0.0,
  "micro_expression_score": 0.0,
  "anomaly_score": 0.0
}
```

**Implementation Notes:**
- **rPPG (remote Photoplethysmography):** For single images, analyze subtle color variation in cheek/forehead regions. For video, extract temporal signal. Single-image rPPG is limited — return low confidence, not zero.
- **Corneal Reflection:** Real eyes show consistent light source reflections in both pupils. GAN eyes often have inconsistent, asymmetric, or missing specular highlights. Detect using blob detection on eye ROI.
- **Micro Expression:** Detect subtle facial muscle activations using optical flow (for video) or texture gradient analysis (for images). 
- Note: This is the hardest agent — implement last. Use stub values (0.5 ± noise) until proper implementation.

**Research Basis:**
- "Detecting Deepfake Videos from Appearance and Behavior" (Mittal et al., 2020)
- "FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals" (Ciftci et al., 2020)

**Key Libraries:** `opencv-python`, `numpy`, `scipy.signal`

---

## 6. VLM Explainability Agent

**Purpose:** Use a Vision-Language Model to generate a natural language assessment and Grad-CAM heatmap highlighting suspicious regions.

**Output Schema:**
```json
{
  "vlm_verdict": "REAL | FAKE | UNCERTAIN",
  "vlm_confidence": 0.0,
  "caption": "string",
  "suspicious_regions": ["string"],
  "gradcam_path": "string",
  "anomaly_score": 0.0
}
```

**Implementation Notes:**
- **BLIP-2:** Use `Salesforce/blip2-opt-2.7b` for image captioning and VQA. Prompt: "Is this a real photograph of a human face or a synthetic/AI-generated image? Describe any artifacts or suspicious regions."
- **Grad-CAM:** Apply to the last convolutional layer of an EfficientNet or XceptionNet model fine-tuned on FaceForensics++. Overlay heatmap on original image and save to disk.
- Parse VLM response to extract verdict and confidence. Use keyword matching as fallback if model is not fine-tuned.
- GPU required for BLIP-2 inference; gracefully degrade to caption-only if GPU unavailable.

**Research Basis:**
- BLIP-2: "BLIP-2: Bootstrapping Language-Image Pre-training" (Li et al., 2023)
- Grad-CAM: "Grad-CAM: Visual Explanations from Deep Networks" (Selvaraju et al., 2017)

**Key Libraries:** `transformers` (BLIP2), `torch`, `torchvision`, `opencv-python`

---

## 7. Fusion Agent

**Purpose:** Combine all agent scores into a final verdict using Bayesian or learned fusion.

**Output Schema:**
```json
{
  "final_score": 0.0,
  "verdict": "REAL | FAKE | UNCERTAIN",
  "confidence": 0.0,
  "method": "bayesian",
  "agent_weights": {
    "geometry": 0.0,
    "frequency": 0.0,
    "texture": 0.0,
    "biological": 0.0,
    "vlm": 0.0
  },
  "anomaly_score": 0.0
}
```

**Implementation Notes:**

**Bayesian Fusion (Phase 1 default):**
```python
import numpy as np

DEFAULT_WEIGHTS = {
    "geometry": 0.15,
    "frequency": 0.25,
    "texture": 0.20,
    "biological": 0.15,
    "vlm": 0.25
}

def bayesian_fusion(scores: dict, weights: dict = DEFAULT_WEIGHTS) -> float:
    weighted_sum = sum(scores[k] * weights[k] for k in weights if k in scores)
    total_weight = sum(weights[k] for k in weights if k in scores)
    return weighted_sum / total_weight if total_weight > 0 else 0.5
```

**Verdict thresholds:**
- `final_score >= 0.65` → FAKE
- `final_score <= 0.35` → REAL  
- Otherwise → UNCERTAIN

**Learned Fusion (Phase 2):**
- Train a 3-layer MLP on agent score vectors using FaceForensics++ labels
- See `references/fusion.md` for training details

---

## 8. Report Agent

**Purpose:** Generate a structured forensic PDF report from the fusion output.

**Output Schema:**
```json
{
  "report_path": "string",
  "report_id": "string",
  "generated_at": "string"
}
```

**Implementation Notes:**
- Use ReportLab to generate PDF
- Include: executive summary, per-agent score table, Grad-CAM heatmap, metadata (EXIF/ELA), final verdict
- Report template structure is TBD — see `references/report-template.md`
- `report_id` format: `DFA-{YYYY}-{random_6_char_hex}`

**Key Libraries:** `reportlab`, `Pillow`
