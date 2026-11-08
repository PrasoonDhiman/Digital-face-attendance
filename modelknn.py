import os
import cv2
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

MODEL_PATH = "model.pkl"



def crop_face_and_embed(bgr_image, detection):

    h, w = bgr_image.shape[:2]

    bbox = detection.location_data.relative_bounding_box

    x1 = int(max(0, bbox.xmin * w))
    y1 = int(max(0, bbox.ymin * h))
    x2 = int(min(w, (bbox.xmin + bbox.width) * w))
    y2 = int(min(h, (bbox.ymin + bbox.height) * h))

    if x2 <= x1 or y2 <= y1:
        return None

    face = bgr_image[y1:y2, x1:x2]

    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    face = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA)

    emb = face.flatten().astype(np.float32) / 255.0

    return emb



def extract_embedding_for_image(stream_or_bytes):

    import mediapipe as mp

    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

    data = stream_or_bytes.read()

    arr = np.frombuffer(data, np.uint8)

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    results = mp_face.process(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )

    if not results.detections:
        return None

    emb = crop_face_and_embed(
        img,
        results.detections[0]
    )

    return emb



def load_model_if_exists():

    if not os.path.exists(MODEL_PATH):
        return None

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_with_model(clf, emb):

    distances, indices = clf.kneighbors([emb], n_neighbors=5)

    label = clf.predict([emb])[0]

    
    conf = 1 / (1 + np.mean(distances))

    return label, float(conf)



def train_model_background(dataset_dir, progress_callback=None):

    import mediapipe as mp

    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

    X = []
    y = []

    
    student_dirs = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    total_students = max(1, len(student_dirs))

    processed = 0

    
    for sid in student_dirs:

        folder = os.path.join(dataset_dir, sid)

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for fn in files:

            path = os.path.join(folder, fn)

            img = cv2.imread(path)

            if img is None:
                print("Could not read:", path)
                continue

            results = mp_face.process(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            )

            if not results.detections:
                print("No face detected:", path)
                continue

            emb = crop_face_and_embed(
                img,
                results.detections[0]
            )

            if emb is None:
                continue

            X.append(emb)

            y.append(int(sid))

        processed += 1

        if progress_callback:

            pct = int((processed / total_students) * 80)

            progress_callback(
                pct,
                f"Processed {processed}/{total_students} students"
            )

    
    if len(X) == 0:

        print("No training data found")

        if progress_callback:
            progress_callback(0, "No training data found")

        return

    
    X = np.stack(X)

    y = np.array(y)

    print("Total Samples:", len(X))
    print("Classes:", np.unique(y))

    # ---- 80-20 Train Test Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---- Train KNN ----
    if progress_callback:
        progress_callback(85, "Training KNN...")

    clf = KNeighborsClassifier(
        n_neighbors=5,
        metric='euclidean'
    )

    clf.fit(X_train, y_train)

    # ---- Predictions ----
    y_pred = clf.predict(X_test)

    # ---- Metrics ----
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    # ---- Print Results ----
    print("\n===== MODEL PERFORMANCE =====")

    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    # ---- Save Model ----
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print("\nModel saved successfully")

    if progress_callback:
        progress_callback(100, "Training complete")


# ---- Main ----
if __name__ == "__main__":

    dataset_dir = "dataset"

    train_model_background(dataset_dir)
