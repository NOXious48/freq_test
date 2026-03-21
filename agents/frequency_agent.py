# MFAD_dev/agents/frequency_agent.py
# Frequency + GAN Sub-Agent (L3) — Deep Sentinel
#
# Architecture: Hybrid FFT math + EfficientViT DL classifier
#
# Methods:
#   A: FFT Radial Spectrum        — Durall et al. CVPR 2020
#   B: Block DCT Coefficients     — Frank et al. ICML 2020
#   C: EfficientViT Inference     — faisalishfaq2005/deepfake-detection-efficientnet-vit
#
# Input : {"input_type": "image", "path": "face_crop_path"}
# Output: 4-field dict matching master agent schema
# {
#   "fft_mid_anomaly_db":  float,  raw dB excess mid-band
#   "fft_high_anomaly_db": float,  raw dB excess high-band
#   "gan_probability":     float,  EfficientViT fake probability [0,1]
#   "anomaly_score":       float   weighted combination [0,1]
# }

import json
import numpy as np
import cv2
from scipy.fftpack import dct as scipy_dct
from pathlib import Path

# ── Load config ───────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(_CONFIG_PATH) as f:
    _CFG = json.load(f)["frequency_agent"]

_FFT_CFG   = _CFG["fft_thresholds"]
_DCT_CFG   = _CFG["dct_thresholds"]
_EN_CFG    = _CFG["efficientnet"]
_WEIGHTS   = _CFG["submethod_weights"]
_IMG_SIZE  = _CFG["image_size"]
_FFT_BANDS = _CFG["fft_bands"]

# ── EfficientViT model (loaded once at import) ───────────────────────────────
_deepfake_model = None
_deepfake_transform = None


def _load_deepfake_model():
    """Load EfficientViT deepfake model once and cache it."""
    global _deepfake_model, _deepfake_transform

    if _deepfake_model is not None:
        return _deepfake_model, _deepfake_transform

    import torch
    from torchvision import transforms
    import sys

    model_dir = Path(__file__).parent.parent / "models" / "deepfake_vit"
    safetensors_path = model_dir / "model.safetensors"

    print(f"[FrequencyAgent] Loading EfficientViT deepfake model...")

    try:
        # Add model dir to path so we can import the model class
        sys.path.insert(0, str(model_dir))
        from model import ImprovedEfficientViT

        model = ImprovedEfficientViT()

        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            model.load_state_dict(state_dict)
            print(f"[FrequencyAgent] Loaded deepfake weights from {safetensors_path}")
        else:
            print(f"[FrequencyAgent] WARNING: No weights at {safetensors_path}")

        model.eval()

        # Model uses mean=0.5, std=0.5 normalization
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        _deepfake_model = model
        _deepfake_transform = transform
        return model, transform

    except Exception as e:
        print(f"[FrequencyAgent] EfficientViT load failed: {e} — using score 0.5")
        return None, None


# ── Agent constants ───────────────────────────────────────────────────────────
STUB_MODE = False

STUB_OUTPUT = {
    "fft_mid_anomaly_db":  0.0,
    "fft_high_anomaly_db": 0.0,
    "gan_probability":     0.0,
    "anomaly_score":       0.0,
}

SCHEMA = STUB_OUTPUT.keys()


def validate_output(output: dict) -> bool:
    """All 4 fields present, float type."""
    if not all(k in output for k in SCHEMA):
        return False
    for k in SCHEMA:
        v = output[k]
        if not isinstance(v, float):
            return False
        if "gan_probability" in k or "anomaly_score" in k:
            if not (0.0 <= v <= 1.0):
                return False
    return True


def run(input: dict) -> dict:
    """
    Frequency + GAN Agent entrypoint.
    Combines FFT spectral analysis with EfficientViT deepfake classifier.

    Args:
        input: {"input_type": "image", "path": str}
    Returns:
        dict with 4 fields
    """
    if STUB_MODE:
        return STUB_OUTPUT.copy()

    image_path = input["path"]

    fft_result = _run_fft_analysis(image_path)
    dct_result = _run_dct_analysis(image_path)
    en_result  = _run_efficientnet_inference(image_path)

    # Weighted anomaly score
    fft_score = float(np.clip(fft_result["fft_mid_anomaly_db"] / 20.0, 0.0, 1.0))
    dct_score = dct_result["dct_score"]
    gan_prob  = en_result["gan_probability"]

    anomaly_score = (
        _WEIGHTS["fft"]          * fft_score +
        _WEIGHTS["dct"]          * dct_score +
        _WEIGHTS["efficientnet"] * gan_prob
    )

    output = {
        "fft_mid_anomaly_db":  fft_result["fft_mid_anomaly_db"],
        "fft_high_anomaly_db": fft_result["fft_high_anomaly_db"],
        "gan_probability":     gan_prob,
        "anomaly_score":       float(np.clip(anomaly_score, 0.0, 1.0)),
    }

    assert validate_output(output), f"Schema validation failed: {output}"
    return output


# ── Method A: FFT Radial Spectrum (Durall et al. CVPR 2020) ──────────────────

def _run_fft_analysis(image_path: str) -> dict:
    """
    Durall et al. 2020 — azimuthal power spectrum analysis.
    Returns raw dB excess values for mid and high frequency bands.
    GAN images show excess energy vs real 1/f roll-off curve.
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

    # Raw dB excess above expected 1/f roll-off
    mid_excess  = float(max(0.0, mid_e  - (low_e + _FFT_CFG["mid_expected_offset"])))
    high_excess = float(max(0.0, high_e - (low_e + _FFT_CFG["high_expected_offset"])))

    return {
        "fft_mid_anomaly_db":  float(np.clip(mid_excess,  0.0, 20.0)),
        "fft_high_anomaly_db": float(np.clip(high_excess, 0.0, 20.0)),
    }


# ── Method B: Block DCT (Frank et al. ICML 2020) ─────────────────────────────

def _run_dct_analysis(image_path: str) -> dict:
    """
    Frank et al. 2020 — Block DCT AC coefficient distribution.
    GAN images show elevated high-frequency AC coefficients.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32)

    bs   = _DCT_CFG["block_size"]
    h, w = img.shape
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
        "dct_score": float(np.clip((hf_ratio - lo) / (hi - lo), 0.0, 1.0)),
    }


# ── Method C: EfficientViT Inference ─────────────────────────────────────────

def _run_efficientnet_inference(image_path: str) -> dict:
    """
    EfficientViT deepfake classifier.
    Single sigmoid output: >0.5 = fake, <=0.5 = real.
    Falls back to 0.5 (neutral) if model unavailable.
    """
    try:
        import torch

        model, transform = _load_deepfake_model()

        if model is None:
            return {"gan_probability": 0.5}

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"gan_probability": 0.5}

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor  = transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            logit    = model(tensor)
            gan_prob = float(torch.sigmoid(logit).item())

        return {"gan_probability": float(np.clip(gan_prob, 0.0, 1.0))}

    except Exception as e:
        print(f"[FrequencyAgent] EfficientViT inference failed: {e} — returning 0.5")
        return {"gan_probability": 0.5}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agents/frequency_agent.py <image_path>")
        sys.exit(1)
    result = run({"input_type": "image", "path": sys.argv[1]})
    print(json.dumps(result, indent=2))
    print("validate_output:", validate_output(result))
