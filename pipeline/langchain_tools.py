# MFAD_dev/pipeline/langchain_tools.py
# Only frequency_tool is active — all other agents are stubs
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
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
