# MFAD_dev/pipeline/langchain_tools.py
# Only frequency_tool is active — all other agents are stubs
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain.tools import tool

@tool
def frequency_tool(face_crop_path: str) -> str:
    """
    Frequency domain analysis on a face crop image.
    Returns JSON: fft_mid_anomaly_db, fft_high_anomaly_db, anomaly_score.
    Techniques: FFT azimuthal averaging (Durall CVPR 2020),
                Block DCT (Frank ICML 2020).
    Band/threshold values are empirically calibrated — see config.json.
    fft fields: float >= 0 (raw dB excess). anomaly_score: float [0, 1].
    Weights: fft=0.55, dct=0.45.
    """
    from agents.frequency_agent import run
    return json.dumps(run({"input_type": "image", "path": face_crop_path}))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline/langchain_tools.py <image_path>")
        sys.exit(1)
    print(frequency_tool.invoke(sys.argv[1]))
