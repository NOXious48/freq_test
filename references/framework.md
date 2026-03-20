# Deep Sentinel — Forensic AI Pipeline Framework

**Version:** 1.0  
**Input Scope:** Single Image (Video: Phase 4)  
**Output Standard:** DFA-2025-TC-00471 PDF Report  
**Orchestration:** LangChain → LangGraph (Phase 3)  
**Fusion:** Bayesian weighted (Phase 1) → MLP (Phase 2)  
**Python:** ≥ 3.9

---

## Pipeline Architecture

```
[Image Input]
      │
      ▼
┌─────────────────────────────────────────┐
│         Preprocessing Agent (L1)        │  ← MANDATORY / RUNS FIRST
│  MediaPipe · dlib · crop · normalize    │
└────────────────────┬────────────────────┘
                     │
         ┌───────────┼──────────────────────────────┐
         │           │           │          │        │
         ▼           ▼           ▼          ▼        ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ ┌──────────────┐
   │Geometry  │ │Frequency │ │ Texture  │ │ Bio  │ │     VLM      │
   │  Agent   │ │  Agent   │ │  Agent   │ │Agent │ │Explainability│
   │   (L2)   │ │   (L3)   │ │   (L4)   │ │ (L5) │ │    (L6)      │
   └────┬─────┘ └────┬─────┘ └────┬─────┘ └──┬───┘ └──────┬───────┘
        │             │             │          │             │
        └─────────────┴─────────────┴──────────┴─────────────┘
                                    │
                                    ▼ anomaly_scores[ ]
                          ┌──────────────────────┐
                          │    Fusion Agent (L7)  │
                          │  Bayesian · MLP Ph2   │
                          └──────────┬────────────┘
                                     │
                                     ▼ final_score · verdict
                          ┌──────────────────────┐
                          │   Report Agent (L8)   │
                          │  ReportLab PDF · ISO  │
                          └──────────────────────┘
```

> All 5 feature agents (L2–L6) run **in parallel** after preprocessing completes.  
> Each agent returns a strict JSON with an `anomaly_score` field ∈ [0, 1].

---

## Project Directory Structure

```
deep_sentinel/
│
├── agents/                         ← one file per agent
│   ├── preprocessing_agent.py
│   ├── geometry_agent.py
│   ├── frequency_agent.py
│   ├── texture_agent.py
│   ├── biological_agent.py
│   ├── vlm_agent.py
│   ├── fusion_agent.py
│   ├── report_agent.py
│   └── __init__.py
│
├── pipeline/                       ← orchestration layer
│   ├── runner.py                   ← sequential orchestrator (Phase 1)
│   ├── parallel.py                 ← ThreadPoolExecutor (Phase 2)
│   ├── langchain_tools.py          ← @tool wrappers for all agents
│   └── langgraph_graph.py          ← StateGraph migration (Phase 3)
│
├── models/                         ← weights & checkpoints
│   ├── efficientnet_ff++.pth       ← EfficientNet-B4 fine-tuned on FF++
│   └── fusion_mlp.pth              ← learned fusion MLP (Phase 2)
│
├── reports/                        ← generated forensic PDFs
│   └── DFA-{YYYY}-TC-{hex}.pdf
│
├── temp/                           ← runtime intermediate files
│   ├── face_crop.jpg
│   ├── normalized.jpg
│   ├── gradcam.jpg
│   └── landmarks.json
│
├── references/                     ← skill reference docs
│   ├── agents.md
│   ├── fusion.md
│   ├── langchain-wiring.md
│   ├── langgraph-migration.md
│   ├── report-template.md
│   └── video-extension.md
│
├── main.py                         ← entrypoint: python main.py image.jpg
├── config.json                     ← weights, thresholds, model paths
└── requirements.txt
```

---

## Agent Catalog

### Agent 01 — Preprocessing (L1) · MANDATORY

**Purpose:** Detect and crop face, extract 468 landmarks via MediaPipe FaceMesh, normalize to 224×224 for downstream DL models. Global dependency — all other agents depend on this output. Fail fast if no face detected.

**Input:**
```json
{ "input_type": "image", "path": "string" }
```

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

**Libraries:** `mediapipe`, `dlib`, `opencv-python`  
**Models:** MediaPipe FaceMesh (468 landmarks), dlib frontal_face_detector (fallback)

---

### Agent 02 — Geometry (L2) · FEATURE

**Purpose:** Detect facial asymmetry and landmark irregularities. GAN encoder-decoder face-swap architectures characteristically fail to preserve natural anthropometric proportions — symmetry index, jaw curvature, interpupillary ratios.

**Input:** `normalized_path` from preprocessing

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

**Libraries:** `dlib` (68-pt), `numpy`, `scipy`  
**Models:** dlib 68-point shape predictor, 3D Morphable Model (3DMM)  
**Research:** Li et al. 2018 — "Exposing DeepFake Videos By Detecting Face Warping Artifacts"

---

### Agent 03 — Frequency (L3) · FEATURE

**Purpose:** Detect GAN upsampling grid artifacts invisible to the human eye. StyleGAN2 exhibits +15.6 dB excess energy at ultra-high spatial frequencies. DCGAN/transposed convolutions leave 4×4 pixel grid signatures in FFT space.

**Input:** `face_crop_path` from preprocessing

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

**Libraries:** `numpy` (fft2, fftshift), `scipy.fftpack`  
**Research:** Durall et al. 2020 — "Unmasking DeepFakes with simple Features"

---

### Agent 04 — Texture (L4) · FEATURE

**Purpose:** GAN-generated faces are hyper-smooth or show seam artifacts at face-body boundary regions. Compares skin texture distribution across 5 facial zones to reference real-face statistics using EMD.

**Input:** `face_crop_path` from preprocessing

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

**Libraries:** `scikit-image` (local_binary_pattern, gabor), `scipy.stats` (wasserstein_distance), `opencv-python`  
**Research:** Rossler et al. 2019 — "FaceForensics++: Learning to Detect Manipulated Facial Images"

---

### Agent 05 — Biological (L5) · COMPLEX

**Purpose:** Detect absence of physiological signals present in real faces. No current GAN replicates cardiac-cycle color variation (rPPG), consistent corneal specular highlights, or authentic subcutaneous vascular patterns.

**Input:** original `image_path` (not crop — needs full color context)

**Output Schema:**
```json
{
  "rppg_score": 0.0,
  "corneal_reflection_score": 0.0,
  "micro_expression_score": 0.0,
  "anomaly_score": 0.0
}
```

**Libraries:** `opencv-python`, `numpy`, `scipy.signal`  
**Note:** Hardest agent — implement last. Use stub (0.5 ± noise) until fully implemented.  
**Research:** Ciftci et al. 2020 — "FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals"

---

### Agent 06 — VLM Explainability (L6) · GPU REQUIRED

**Purpose:** Vision-Language Model generates natural-language forensic assessment and Grad-CAM pixel-level saliency map highlighting suspicious regions for the court report.

**Input:** `face_crop_path` from preprocessing

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

**Libraries:** `transformers` (BLIP-2), `torch`, `torchvision`, `opencv-python`  
**Models:** `Salesforce/blip2-opt-2.7b`, EfficientNet-B4 (FF++ fine-tuned)  
**Research:** Li et al. 2023 — "BLIP-2"; Selvaraju et al. 2017 — "Grad-CAM"

---

### Agent 07 — Fusion (L7) · ENSEMBLE

**Purpose:** Combines all 5 feature agent `anomaly_score` values into a final DeepFake Prediction Score using Bayesian weighted fusion with isotonic regression calibration.

**Input:** dict of all agent outputs (extracts `anomaly_score` from each)

**Output Schema:**
```json
{
  "final_score": 0.0,
  "verdict": "REAL | FAKE | UNCERTAIN",
  "confidence": 0.0,
  "method": "bayesian | learned",
  "agent_weights": {
    "geometry": 0.15,
    "frequency": 0.25,
    "texture": 0.20,
    "biological": 0.15,
    "vlm": 0.25
  },
  "anomaly_score": 0.0
}
```

**Verdict thresholds:**
- `final_score >= 0.65` → **FAKE**
- `final_score <= 0.35` → **REAL**
- `0.35 < score < 0.65` → **UNCERTAIN**

---

### Agent 08 — Report (L8) · PDF OUTPUT

**Purpose:** Generate a court-grade forensic PDF matching the DFA-2025-TC-00471 standard. Must satisfy ISO/IEC 27037, SWGDE Best Practices, NIST SP 800-101r1.

**Input:** full fusion output + vlm caption + all agent outputs

**Output Schema:**
```json
{
  "report_path": "string",
  "report_id": "DFA-{YYYY}-TC-{6_char_hex}",
  "generated_at": "ISO8601 string"
}
```

**Libraries:** `reportlab`, `Pillow`

---

## Master I/O Schema

**Pipeline Input:**
```json
{
  "input_type": "image",
  "path": "/absolute/path/to/image.jpg"
}
```

**Master Output:**
```json
{
  "preprocessing": {},
  "geometry": {},
  "frequency": {},
  "texture": {},
  "biological": {},
  "vlm": {},
  "fusion": {
    "final_score": 0.0,
    "verdict": "REAL | FAKE | UNCERTAIN",
    "confidence": 0.0,
    "method": "bayesian | learned"
  },
  "report_path": "reports/DFA-2025-TC-a3f1b2.pdf"
}
```

---

## Fusion Weight Configuration

| Agent | Default Weight | Rationale |
|---|---|---|
| VLM Explainability | 0.25 | Semantic + pixel-level combined signal |
| Frequency | 0.25 | Most reliable GAN artifact detector |
| Texture | 0.20 | Strong at seam detection |
| Geometry | 0.15 | Fast, no DL, moderate reliability |
| Biological | 0.15 | Hard to implement, lower initial confidence |
| **Total** | **1.00** | |

**Calibration target:** AUC-ROC ≥ 0.983, FPR < 2.1% at 90% recall (reference: DFA-2025-TC-00471)

---

## Technology Stack

| Layer | Libraries / Models |
|---|---|
| Orchestration | LangChain, LangGraph (Phase 3), ThreadPoolExecutor |
| Face Detection | MediaPipe FaceMesh, dlib, RetinaFace, MTCNN |
| Signal Analysis | numpy.fft2, scipy.fftpack DCT, scikit-image LBP, Gabor bank, scipy.stats wasserstein |
| Deep Learning | EfficientNet-B4 (FF++ fine-tuned), XceptionNet, BLIP-2, Grad-CAM, PyTorch, transformers |
| Biological | rPPG color analysis, cv2 blob detection, scipy.signal |
| Metadata Forensics | piexif, ELA, PRNU, SHA-256 / MD5 |
| Report Generation | ReportLab, Pillow |
| Training Datasets | FaceForensics++ v3, DFDC, Celeb-DF v2 |

---

## Build Roadmap

### Phase 1 — Scaffold (Current)

1. Generate all 8 agents as **stubs** — valid JSON, zeroed values, `STUB_MODE = True`
2. Wire stubs with LangChain `@tool` wrappers + sequential `run_pipeline()` orchestrator
3. Run end-to-end on test image — confirm full master JSON output is produced
4. Commit baseline — all stubs passing `validate_output()`

### Phase 2 — Implement (Agent by Agent)

Replace stubs in this order (dependency-driven):

1. **Preprocessing** — global dependency, everything blocks on it
2. **Geometry** — no DL, fast to validate
3. **Frequency** — numpy/scipy only, easy to unit test
4. **Texture** — scikit-image, moderate complexity
5. **Biological** — hardest, implement last in feature group
6. **VLM Explainability** — requires GPU, BLIP-2 inference
7. **Fusion** — implement after all feature agents are non-stub
8. **Report** — implement last, depends on all outputs

### Phase 3 — Upgrade

1. LangGraph migration — `PipelineState` TypedDict + `StateGraph` (see `langgraph-migration.md`)
2. Learned Fusion — train 3-layer MLP on FF++ agent score vectors (see `fusion.md`)
3. Parallel execution — `ThreadPoolExecutor(max_workers=5)` for feature agents
4. Video extension — temporal consistency + full rPPG signal (see `video-extension.md`, out of scope now)

---

## Code Generation Rules (All Agents)

Every agent file must follow this skeleton:

```python
# agents/{name}_agent.py

STUB_MODE = True  # Set to False when implementing

STUB_OUTPUT = {
    # all required output fields with zero/default values
}

SCHEMA = STUB_OUTPUT.keys()

def validate_output(output: dict) -> bool:
    return all(k in output for k in SCHEMA)

def run(input: dict) -> dict:
    """
    {AgentName} Agent: [what it detects and why it indicates deepfake]
    
    Args:
        input: dict with keys input_type, path (and optionally landmarks_path, etc.)
    Returns:
        dict matching SCHEMA with anomaly_score in [0, 1]
    """
    if STUB_MODE:
        return STUB_OUTPUT.copy()
    
    # Real implementation here
    ...
```

---

## Reference Report Alignment

Each agent maps directly to a section of the DFA-2025-TC-00471 forensic report:

| Agent | Report Section | Key Evidence from Reference |
|---|---|---|
| Preprocessing | §1 Chain of Custody | SHA-256 integrity, face bbox, image metadata |
| Geometry | §5.1 Facial Geometry | Symmetry 0.74 (−19.6%), jaw curvature 11.2° (+124%), ear alignment 8.7px |
| Frequency | §5.2 GAN Artefact & Frequency | Mid-freq +9.4 dB (p<0.001), ultra-high +15.6 dB — StyleGAN2 signature |
| Texture | §5.3 Texture Consistency | Neck-face EMD 0.274 (3.4σ), cheek-jaw LBP 0.58 — seam artifact |
| Biological | §5.5 Biological Plausibility | rPPG SNR 0.09 (authentic >0.45), corneal deviation 14.3°, vascular r=0.41 |
| VLM | §5.4 Explainability Heatmap | RED zone (eyes/nose/mouth) salience 0.91, GAN prob 0.93 |
| Fusion | §6 Confidence Scoring | Final 95.0%, 95% CI [93.1%–96.6%], ECE 0.014 |
| Report | §8–§10 Legal Certification | ISO/IEC 27037, SWGDE, NIST SP 800-101r1, SHA-256 chain |

---

## Per-Agent Score Reference (DFA-2025-TC-00471)

| Module | Score |
|---|---|
| Facial Geometry & Landmark Deviation | 88.4% |
| GAN Artefact Detection (EfficientNet-B4) | 96.7% |
| Frequency-Domain Spectral Anomaly | 91.2% |
| Texture / Skin-Tone Consistency | 89.5% |
| VLM Explainability Attention Score | 93.1% |
| Biological Plausibility Failure | 82.6% |
| Metadata & Provenance Anomalies | 97.3% |
| **Ensemble Bayesian Fusion (FINAL)** | **95.0%** |

---

## Config Schema (config.json)

```json
{
  "fusion_weights": {
    "geometry": 0.15,
    "frequency": 0.25,
    "texture": 0.20,
    "biological": 0.15,
    "vlm": 0.25
  },
  "verdict_thresholds": {
    "fake": 0.65,
    "real": 0.35
  },
  "models": {
    "efficientnet": "models/efficientnet_ff++.pth",
    "fusion_mlp": "models/fusion_mlp.pth",
    "blip2": "Salesforce/blip2-opt-2.7b"
  },
  "output": {
    "report_dir": "reports/",
    "temp_dir": "temp/",
    "report_id_prefix": "DFA"
  },
  "stub_mode": true
}
```

---

## Requirements

```
# Core
numpy>=1.24
scipy>=1.10
opencv-python>=4.8
scikit-image>=0.21
Pillow>=10.0

# Face detection
mediapipe>=0.10
dlib>=19.24

# Deep learning
torch>=2.0
torchvision>=0.15
transformers>=4.35

# Orchestration
langchain>=0.1
langgraph>=0.0.40

# Report
reportlab>=4.0

# Metadata forensics
piexif>=1.1
exifread>=3.0

# Optional
ollama  # local LLM alternative to BLIP-2
```
