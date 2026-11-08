import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from model import load_model_if_exists, crop_face_and_embed
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# =========================
# Load trained model
# =========================
clf = load_model_if_exists()

if clf is None:
    print("❌ Model not found. Train model first.")
    exit()

print("✅ Model loaded successfully")

# =========================
# Dataset path
# =========================
DATASET_DIR = "dataset"

y_true = []
y_pred = []

# =========================
# Loop dataset
# =========================
for sid in os.listdir(DATASET_DIR):
    folder = os.path.join(DATASET_DIR, sid)

    if not os.path.isdir(folder):
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # =========================
        # Face detection (SAFE VERSION)
        # =========================
        import mediapipe as mp

        mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.3
        )

        results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.detections:
            print(f"⚠ No face detected: {img_path}")
            continue

        emb = crop_face_and_embed(img, results.detections[0])

        if emb is None:
            continue

        # =========================
        # Prediction
        # =========================
        proba = clf.predict_proba([emb])[0]
        idx = np.argmax(proba)

        pred_label = clf.classes_[idx]

        # =========================
        # Ground truth
        # IMPORTANT: keep as STRING (safe)
        # =========================
        y_true.append(sid)
        y_pred.append(str(pred_label))

# =========================
# Final checks
# =========================
print("\nTotal samples:", len(y_true))

if len(y_true) == 0:
    print("❌ No valid samples found. Check dataset or face detection.")
    exit()

# =========================
# Metrics
# =========================
print("\nAccuracy:", accuracy_score(y_true, y_pred) * 100)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# =========================
# Confusion Matrix
# =========================
labels = sorted(list(set(y_true + y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(12, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Face Recognition Confusion Matrix")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()