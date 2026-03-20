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
