import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# DATASET PATH
# ---------------------------------------------------
DATASET_PATH = r"C:\Users\dhima\Documents\Digital-Facial-Recognisation-Attendance-System\dataset"

# ---------------------------------------------------
# MEDIAPIPE SETUP
# ---------------------------------------------------
mp_face = mp.solutions.face_detection

face_detection = mp_face.FaceDetection(
    min_detection_confidence=0.5
)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
X = []
y = []

print("[INFO] Loading dataset...")

for person_name in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        frame = cv2.imread(image_path)

        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb)

        if results.detections:

            detection = results.detections[0]

            bbox = detection.location_data.relative_bounding_box

            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            face_crop = frame[y1:y1+bh, x:x+bw]

            if face_crop.size == 0:
                continue

            # Resize
            face_crop = cv2.resize(face_crop, (128, 128))

            # Normalize
            face_crop = face_crop / 255.0

            # Create embedding
            embedding = face_crop.flatten()

            # Use first 128 values
            embedding = embedding[:128]

            X.append(embedding)
            y.append(person_name)

print(f"[INFO] Total samples loaded: {len(X)}")

# Convert to numpy
X = np.array(X)
y = np.array(y)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# OOB ERROR CURVE
# ---------------------------------------------------
estimators = [10, 50, 100, 150, 200, 250, 300]

oob_errors = []

print("\n[INFO] Training models...")

for n in estimators:

    print(f"Training with {n} trees...")

    model = RandomForestClassifier(
        n_estimators=n,
        oob_score=True,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    oob_error = 1 - model.oob_score_

    oob_errors.append(oob_error)

    print(f"OOB Error: {oob_error:.4f}")

# ---------------------------------------------------
# PLOT CURVE
# ---------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    estimators,
    oob_errors,
    marker='o',
    linewidth=2
)

plt.xlabel(
    "Number of Trees",
    fontsize=12,
    fontstyle='italic'
)

plt.ylabel(
    "OOB Error",
    fontsize=12,
    fontstyle='italic'
)

plt.title(
    "Out-of-Bag Error Curve for Random Forest",
    fontsize=14,
    fontstyle='italic'
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "oob_error_curve.png",
    dpi=300
)

plt.show()

print("\n[INFO] OOB Error Curve saved successfully")
