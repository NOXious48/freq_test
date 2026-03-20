# LangChain Wiring Reference

## Full Pipeline Wiring with LangChain Tools

This file shows how to wire all agents as LangChain tools and build the executor.

### Tool Registration Pattern

```python
from langchain.tools import tool
from langchain_core.messages import HumanMessage
import json

# Import all agents
from agents.preprocessing_agent import run as run_preprocessing
from agents.geometry_agent import run as run_geometry
from agents.frequency_agent import run as run_frequency
from agents.texture_agent import run as run_texture
from agents.biological_agent import run as run_biological
from agents.vlm_agent import run as run_vlm
from agents.fusion_agent import run as run_fusion
from agents.report_agent import run as run_report

@tool
def preprocessing_tool(image_path: str) -> str:
    """Preprocess image: detect face, extract landmarks, normalize. MUST run first."""
    result = run_preprocessing({"input_type": "image", "path": image_path})
    return json.dumps(result)

@tool
def geometry_tool(normalized_path: str) -> str:
    """Analyze facial geometry for asymmetry and landmark irregularities."""
    result = run_geometry({"input_type": "image", "path": normalized_path})
    return json.dumps(result)

@tool
def frequency_tool(face_crop_path: str) -> str:
    """Analyze frequency domain for GAN artifacts (FFT, DCT)."""
    result = run_frequency({"input_type": "image", "path": face_crop_path})
    return json.dumps(result)

@tool
def texture_tool(face_crop_path: str) -> str:
    """Analyze skin texture using LBP, Gabor filters, and EMD."""
    result = run_texture({"input_type": "image", "path": face_crop_path})
    return json.dumps(result)

@tool
def biological_tool(image_path: str) -> str:
    """Analyze biological signals: rPPG and corneal reflections."""
    result = run_biological({"input_type": "image", "path": image_path})
    return json.dumps(result)

@tool
def vlm_tool(face_crop_path: str) -> str:
    """Run VLM (BLIP-2) analysis and generate Grad-CAM heatmap."""
    result = run_vlm({"input_type": "image", "path": face_crop_path})
    return json.dumps(result)

@tool
def fusion_tool(agent_outputs_json: str) -> str:
    """Fuse all agent scores into a final verdict."""
    agent_outputs = json.loads(agent_outputs_json)
    result = run_fusion(agent_outputs)
    return json.dumps(result)

@tool
def report_tool(fusion_output_json: str) -> str:
    """Generate forensic PDF report from fusion output."""
    fusion_output = json.loads(fusion_output_json)
    result = run_report(fusion_output)
    return json.dumps(result)
```

### Pipeline Orchestrator (Non-Agent, Sequential)

For deterministic pipelines, prefer a simple sequential orchestrator over a ReAct agent:

```python
import json
from pathlib import Path

def run_pipeline(image_path: str) -> dict:
    """
    Run the full deepfake detection pipeline on a single image.
    Returns the master output schema.
    """
    print(f"[Pipeline] Starting analysis: {image_path}")
    
    # Step 1: Preprocessing (mandatory)
    print("[1/7] Preprocessing...")
    preprocessing_out = run_preprocessing({"input_type": "image", "path": image_path})
    
    if not preprocessing_out.get("face_detected"):
        return {"error": "No face detected in image", "preprocessing": preprocessing_out}
    
    face_crop = preprocessing_out["face_crop_path"]
    normalized = preprocessing_out["normalized_path"]
    
    # Step 2: Run all feature agents in parallel (or sequential for simplicity)
    print("[2/7] Geometry analysis...")
    geometry_out = run_geometry({"input_type": "image", "path": normalized})
    
    print("[3/7] Frequency analysis...")
    frequency_out = run_frequency({"input_type": "image", "path": face_crop})
    
    print("[4/7] Texture analysis...")
    texture_out = run_texture({"input_type": "image", "path": face_crop})
    
    print("[5/7] Biological analysis...")
    biological_out = run_biological({"input_type": "image", "path": image_path})
    
    print("[6/7] VLM explainability...")
    vlm_out = run_vlm({"input_type": "image", "path": face_crop})
    
    # Step 3: Fusion
    print("[7a/7] Fusing scores...")
    agent_scores = {
        "geometry": geometry_out.get("anomaly_score", 0.0),
        "frequency": frequency_out.get("anomaly_score", 0.0),
        "texture": texture_out.get("anomaly_score", 0.0),
        "biological": biological_out.get("anomaly_score", 0.0),
        "vlm": vlm_out.get("anomaly_score", 0.0),
    }
    fusion_out = run_fusion(agent_scores)
    
    # Step 4: Report
    print("[7b/7] Generating report...")
    report_out = run_report({**fusion_out, "vlm_caption": vlm_out.get("caption", "")})
    
    # Assemble master output
    master_output = {
        "preprocessing": preprocessing_out,
        "geometry": geometry_out,
        "frequency": frequency_out,
        "texture": texture_out,
        "biological": biological_out,
        "vlm": vlm_out,
        "fusion": fusion_out,
        "report_path": report_out.get("report_path", "")
    }
    
    print(f"[Pipeline] Done. Verdict: {fusion_out.get('verdict')} "
          f"(score: {fusion_out.get('final_score', 0):.3f})")
    return master_output


if __name__ == "__main__":
    import sys
    result = run_pipeline(sys.argv[1])
    print(json.dumps(result, indent=2))
```

### Parallel Execution with ThreadPoolExecutor

When feature agents are mature enough, run them in parallel:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_feature_agents_parallel(preprocessing_out: dict) -> dict:
    face_crop = preprocessing_out["face_crop_path"]
    normalized = preprocessing_out["normalized_path"]
    original = preprocessing_out.get("original_path")
    
    tasks = {
        "geometry": (run_geometry, {"input_type": "image", "path": normalized}),
        "frequency": (run_frequency, {"input_type": "image", "path": face_crop}),
        "texture": (run_texture, {"input_type": "image", "path": face_crop}),
        "biological": (run_biological, {"input_type": "image", "path": original}),
        "vlm": (run_vlm, {"input_type": "image", "path": face_crop}),
    }
    
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fn, inp): name for name, (fn, inp) in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {"error": str(e), "anomaly_score": 0.5}
    
    return results
```
