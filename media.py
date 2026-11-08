import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------
# IMAGE PATH
# ---------------------------------------------------
IMAGE_PATH = r"C:\Users\dhima\Documents\Digital-Facial-Recognisation-Attendance-System\dataset\29\1778726938.344402_15.jpg"

# ---------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------
frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print("Image not found")
    exit()

print("[INFO] Image loaded successfully")

# ---------------------------------------------------
# 1. ORIGINAL IMAGE
# ---------------------------------------------------
cv2.imwrite("1_raw_image.jpg", frame)

# ---------------------------------------------------
# 2. RGB CONVERSION
# ---------------------------------------------------
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(5, 5))
plt.imshow(rgb)
plt.axis("off")
plt.title(
    "RGB Image",
    fontsize=12,
    fontstyle='italic'
)

plt.savefig("2_rgb_image.jpg", bbox_inches='tight')
plt.close()

# ---------------------------------------------------
# MEDIAPIPE FACE DETECTION
# ---------------------------------------------------
mp_face = mp.solutions.face_detection

face_detection = mp_face.FaceDetection(
    min_detection_confidence=0.5
)

results = face_detection.process(rgb)

if results.detections:

    print("[INFO] Face detected successfully")

    for detection in results.detections:

        bbox = detection.location_data.relative_bounding_box

        h, w, _ = frame.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # ---------------------------------------------------
        # 3. FACE DETECTION OUTPUT
        # ---------------------------------------------------
        detected_frame = frame.copy()

        cv2.rectangle(
            detected_frame,
            (x, y),
            (x + bw, y + bh),
            (0, 255, 0),
            3
        )

        cv2.putText(
            detected_frame,
            "Detected Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imwrite("3_detected_face.jpg", detected_frame)

        # ---------------------------------------------------
        # 4. CROPPED FACE
        # ---------------------------------------------------
        face_crop = frame[y:y+bh, x:x+bw]

        cv2.imwrite("4_cropped_face.jpg", face_crop)

        # ---------------------------------------------------
        # 5. RESIZED FACE
        # ---------------------------------------------------
        resized_face = cv2.resize(face_crop, (128, 128))

        cv2.imwrite("5_resized_face.jpg", resized_face)

        # ---------------------------------------------------
        # 6. NORMALIZATION
        # ---------------------------------------------------
        normalized_face = resized_face / 255.0

        norm_display = (normalized_face * 255).astype(np.uint8)

        cv2.imwrite("6_normalized_face.jpg", norm_display)

        # ---------------------------------------------------
        # 7. FEATURE EXTRACTION
        # ---------------------------------------------------
        embedding = normalized_face.flatten().astype(np.float32)

        # First 128 feature values
        embedding = embedding[:128]

        # ---------------------------------------------------
        # SAVE FEATURE VALUES
        # ---------------------------------------------------
        df = pd.DataFrame({
            "Feature Index": np.arange(len(embedding)),
            "Feature Value": embedding
        })

        df.to_csv("sample_feature_values3.csv", index=False)

        print("\nSample Feature Values:")
        print(df.head(10))

        # ---------------------------------------------------
        # 8. EMBEDDING GRAPH
        # ---------------------------------------------------
        plt.figure(figsize=(10, 4))

        plt.plot(
            embedding,
            linewidth=2
        )

        plt.title(
            "Extracted Face Embedding Features",
            fontsize=16,
            fontstyle='italic'
        )

        plt.xlabel(
            "Feature Index",
            fontsize=12,
            fontstyle='italic'
        )

        plt.ylabel(
            "Feature Value",
            fontsize=12,
            fontstyle='italic'
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            "7_embedding_graph.png",
            dpi=300
        )

        plt.close()

        # ---------------------------------------------------
        # 9. COMBINED PREPROCESSING FIGURE
        # ---------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        images = [
            ("1_raw_image.jpg", "Original Input Image"),
            ("2_rgb_image.jpg", "RGB Color Conversion"),
            ("3_detected_face.jpg", "Face Detection using MediaPipe"),
            ("4_cropped_face.jpg", "Extracted Face Region"),
            ("5_resized_face.jpg", "Resized Face Image"),
            ("6_normalized_face.jpg", "Normalized Face Image")
        ]

        for ax, (img_path, title) in zip(axes.flatten(), images):

            img = cv2.imread(img_path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax.imshow(img)

            ax.set_title(
                title,
                fontsize=12,
                fontstyle='italic',
                pad=10
            )

            ax.axis("off")

        # ---------------------------------------------------
        # MAIN FIGURE TITLE
        # ---------------------------------------------------
        fig.suptitle(
            "Preprocessing Stages in the Proposed Face Recognition System",
            fontsize=18,
            fontstyle='italic',
            y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(
            "preprocessing_steps_figure3.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print("\n[INFO] ALL FILES GENERATED SUCCESSFULLY")
        print("------------------------------------------------")
        print("1. preprocessing_steps_figure.png")
        print("2. 7_embedding_graph.png")
        print("3. sample_feature_values.csv")
        print("------------------------------------------------")

        break

else:
    print("No face detected")

# ---------------------------------------------------
# CLEANUP
# ---------------------------------------------------
face_detection.close()
cv2.destroyAllWindows()
