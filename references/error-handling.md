# Error Handling Specification — Deep Sentinel Pipeline

**Purpose:** Defines how every failure mode is handled across all agents and the orchestrator.
The pipeline must never crash silently — every failure must be logged, recovered from gracefully,
and surfaced in the final report.

---

## Core Philosophy

The pipeline distinguishes between two failure classes:

**Fatal failure** — stops the pipeline immediately. Only one case qualifies: Preprocessing fails
to detect a face. Without a face crop, no feature agent can run. Return an error immediately.

**Recoverable failure** — an individual feature agent crashes or times out. The pipeline
continues with a neutral score (0.5) for that agent. The final report flags the failure.

---

## Error Output Schema

When any agent fails, it returns this standardised error dict instead of its normal output:

```json
{
  "error": "ExceptionType: human-readable message",
  "anomaly_score": 0.5,
  "agent": "frequency",
  "failed_at": "ISO8601 timestamp"
}
```

The `anomaly_score` of 0.5 is deliberately neutral — it does not push the fusion result
toward FAKE or REAL. It simply reduces the effective weight of that agent in the ensemble.

---

## Preprocessing Agent — Fatal Failures

Preprocessing is the only agent whose failure halts the entire pipeline.

```python
def run(input: dict) -> dict:
    try:
        # ... face detection logic ...
        if not face_detected:
            return {
                "face_detected": False,
                "error": "No face detected in the submitted image",
                "anomaly_score": 0.0,
                "face_crop_path": None,
                "normalized_path": None,
                "landmarks_path": None,
                "original_path": input.get("path"),
                "bbox": []
            }
    except Exception as e:
        return {
            "face_detected": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "anomaly_score": 0.0,
            "face_crop_path": None,
            "normalized_path": None,
            "landmarks_path": None,
            "original_path": input.get("path"),
            "bbox": []
        }
```

The orchestrator (`pipeline/runner.py`) checks `face_detected` immediately after preprocessing.
If False, it returns:

```python
if not preprocessing_out.get("face_detected"):
    return {
        "error": "Pipeline halted — no face detected",
        "preprocessing": preprocessing_out,
        "geometry": None,
        "frequency": None,
        "texture": None,
        "biological": None,
        "vlm": None,
        "metadata": None,
        "fusion": None,
        "report_path": None
    }
```

---

## Feature Agents — Recoverable Failures

All feature agents (Geometry, Frequency, Texture, Biological, VLM, Metadata) use this wrapper:

```python
import traceback
from datetime import datetime, timezone

def safe_run(agent_module, input: dict, agent_name: str) -> dict:
    """
    Wraps any agent's run() in a try/except.
    On failure, returns a neutral error dict instead of crashing the pipeline.
    """
    try:
        result = agent_module.run(input)
        if not agent_module.validate_output(result):
            raise ValueError(f"validate_output() failed — missing schema fields")
        return result
    except Exception as e:
        print(f"[ERROR] {agent_name} failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return {
            "error": f"{type(e).__name__}: {str(e)}",
            "anomaly_score": 0.5,
            "agent": agent_name,
            "failed_at": datetime.now(timezone.utc).isoformat()
        }
```

Usage in the orchestrator:

```python
from pipeline.error_handling import safe_run
from agents import frequency_agent, geometry_agent  # etc.

frequency_out  = safe_run(frequency_agent,  {"input_type": "image", "path": face_crop}, "frequency")
geometry_out   = safe_run(geometry_agent,   {"input_type": "image", "path": normalized}, "geometry")
```

---

## Fusion Agent — Handling Missing / Error Scores

The fusion agent must handle the case where any agent returned an error dict.

```python
def extract_score(agent_output: dict, agent_name: str) -> float:
    """
    Safely extracts anomaly_score from an agent output.
    If the agent failed, returns 0.5 (neutral) and logs a warning.
    """
    if "error" in agent_output:
        print(f"[FUSION WARNING] {agent_name} returned error — using neutral score 0.5")
        return 0.5
    score = agent_output.get("anomaly_score", 0.5)
    # Clamp to valid range in case of floating point edge case
    return float(max(0.0, min(1.0, score)))
```

Fusion always runs regardless of how many agents failed. Even if 4 of 6 agents fail,
the fusion produces a verdict from the remaining 2. The report will clearly flag this situation.

---

## Timeout Handling

Long-running agents (VLM, Biological) should be wrapped with a timeout:

```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Agent execution timed out")

def run_with_timeout(agent_module, input: dict, agent_name: str, timeout_seconds: int = 120) -> dict:
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = safe_run(agent_module, input, agent_name)
        signal.alarm(0)  # cancel alarm
        return result
    except TimeoutError:
        signal.alarm(0)
        print(f"[TIMEOUT] {agent_name} exceeded {timeout_seconds}s — using neutral score")
        return {
            "error": f"TimeoutError: exceeded {timeout_seconds}s execution limit",
            "anomaly_score": 0.5,
            "agent": agent_name,
            "failed_at": datetime.now(timezone.utc).isoformat()
        }
```

**Default timeouts:**

| Agent | Timeout |
|---|---|
| Preprocessing | 30s |
| Geometry | 15s |
| Frequency | 30s |
| Texture | 30s |
| Biological | 60s |
| VLM Explainability | 120s |
| Metadata | 15s |
| Fusion | 10s |
| Report | 60s |

---

## Temp File Cleanup

On any pipeline failure (after preprocessing succeeds), temp files must be cleaned up:

```python
import shutil
import os

def cleanup_temp(temp_dir: str = "temp/"):
    """Remove all intermediate files from a pipeline run."""
    for f in ["face_crop.jpg", "normalized.jpg", "gradcam.jpg", "landmarks.json"]:
        path = os.path.join(temp_dir, f)
        if os.path.exists(path):
            os.remove(path)
```

The orchestrator calls `cleanup_temp()` in a `finally` block:

```python
try:
    result = run_pipeline(image_path)
finally:
    cleanup_temp()
```

---

## Error Logging

All errors are written to `logs/pipeline_{date}.log` in this format:

```
[2025-11-14 09:42:31 UTC] [ERROR] frequency | FileNotFoundError: face_crop.jpg not found
[2025-11-14 09:42:31 UTC] [ERROR] frequency | Traceback: ...
[2025-11-14 09:42:31 UTC] [FUSION] frequency assigned neutral score 0.5
[2025-11-14 09:43:01 UTC] [DONE] verdict=FAKE final_score=0.873 report=reports/DFA-2025-TC-a3b1f2.pdf
```

---

## Report — Error Flagging

The Report Agent must surface any agent failures in the PDF:

If any agent returned an error, the per-agent score table in the report shows:

| Module | Score | Status |
|---|---|---|
| Frequency Domain | N/A | ⚠ AGENT ERROR — neutral score applied |
| Texture Consistency | 89.5% | ✓ |

The executive summary includes a one-line note:
"Note: 1 of 6 detection modules failed during analysis. Score may underestimate manipulation
confidence. See §4 for details."

---

## Input Validation

Before any agent runs, the orchestrator validates the input image:

```python
import os
import cv2

def validate_input(image_path: str) -> dict:
    """
    Validates the input image before pipeline execution.
    Returns {"valid": True} or {"valid": False, "error": "..."}
    """
    if not os.path.exists(image_path):
        return {"valid": False, "error": f"File not found: {image_path}"}

    if not image_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        return {"valid": False, "error": "Unsupported file format — use JPEG, PNG, BMP, or WebP"}

    img = cv2.imread(image_path)
    if img is None:
        return {"valid": False, "error": "OpenCV could not read the image — file may be corrupt"}

    h, w = img.shape[:2]
    if h < 64 or w < 64:
        return {"valid": False, "error": f"Image too small ({w}×{h}) — minimum 64×64 px required"}

    return {"valid": True}
```
