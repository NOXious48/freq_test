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
        "high_freq_energy":  float(np.clip(high_e / 100.0, 0.0, 1.0)),
        "mid_freq_ratio":    float(np.clip(mid_e / (low_e + 1e-8), 0.0, 1.0)),
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

    # Trim to common shape: pg_y is (N-1, M), pg_x is (N, M-1)
    min_rows = min(pg_y.shape[0], pg_x.shape[0])
    min_cols = min(pg_y.shape[1], pg_x.shape[1])
    phase_disc = np.sqrt(pg_y[:min_rows, :min_cols]**2 + pg_x[:min_rows, :min_cols]**2).mean()

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
