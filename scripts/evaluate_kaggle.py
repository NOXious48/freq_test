# scripts/evaluate_kaggle.py
import sys, os, tempfile
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from agents.frequency_agent import run, validate_output

REAL_DIR = Path("test_images/Dataset/Test/Real")
FAKE_DIR = Path("test_images/Dataset/Test/Fake")
THRESHOLD = 0.50

real_images = list(REAL_DIR.glob("*.jpg")) + list(REAL_DIR.glob("*.png"))
fake_images = list(FAKE_DIR.glob("*.jpg")) + list(FAKE_DIR.glob("*.png"))

print(f"Real: {len(real_images)}  Fake: {len(fake_images)}")

y_true, y_pred, y_scores = [], [], []
face_crop_rate = []

for img_path, label in [(p, 0) for p in real_images] + [(p, 1) for p in fake_images]:
    try:
        result = run({"input_type": "image", "path": str(img_path)})
        score  = result["anomaly_score"]
        pred   = 1 if score >= THRESHOLD else 0
        y_true.append(label)
        y_pred.append(pred)
        y_scores.append(score)
        face_crop_rate.append(result.get("face_cropped", False))
    except Exception as e:
        print(f"  ERROR {img_path.name}: {e}")

y_true  = np.array(y_true)
y_pred  = np.array(y_pred)
y_scores = np.array(y_scores)

TP = ((y_pred == 1) & (y_true == 1)).sum()
TN = ((y_pred == 0) & (y_true == 0)).sum()
FP = ((y_pred == 1) & (y_true == 0)).sum()
FN = ((y_pred == 0) & (y_true == 1)).sum()

from sklearn.metrics import roc_auc_score
print(f"\n{'='*50}")
print(f"  Accuracy    : {(TP+TN)/len(y_true):.4f}")
print(f"  Recall      : {TP/(TP+FN):.4f}  (fake detection rate)")
print(f"  Specificity : {TN/(TN+FP):.4f}  (real rejection rate)")
print(f"  Precision   : {TP/(TP+FP):.4f}")
print(f"  AUC-ROC     : {roc_auc_score(y_true, y_scores):.4f}")
print(f"  Face cropped: {sum(face_crop_rate)}/{len(face_crop_rate)} ({100*sum(face_crop_rate)/len(face_crop_rate):.1f}%)")
print(f"\n  Confusion Matrix:")
print(f"                   Predicted Real | Predicted Fake")
print(f"  Actual Real    |      {TN:<10}  |      {FP}")
print(f"  Actual Fake    |      {FN:<10}  |      {TP}")
print(f"{'='*50}")
