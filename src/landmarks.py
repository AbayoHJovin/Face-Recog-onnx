# src/landmarks.py
"""
Minimal pipeline:
camera -> Haar face box -> MediaPipe FaceMesh (full-frame) -> extract 5 keypoints -> draw
Execute:
python -m src.landmarks
Controls:
q : exit
"""

import cv2
import numpy as np
import mediapipe as mp

# 5-point indices (FaceMesh)
IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291


def main():
    # Haar
    classifier_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_classifier = cv2.CascadeClassifier(classifier_path)
    if face_classifier.empty():
        raise RuntimeError(f"Failed to load cascade: {classifier_path}")

    # FaceMesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")

    print("Haar + FaceMesh 5pt (minimal). Press 'q' to quit.")

    while True:
        success, img = camera.read()
        if not success:
            break

        H, W = img.shape[:2]
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = face_classifier.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        # draw ALL haar faces (no ranking)
        for x, y, w, h in detected_faces:
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

        # FaceMesh on full frame (simple)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_image)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            indices = [
                IDX_LEFT_EYE,
                IDX_RIGHT_EYE,
                IDX_NOSE_TIP,
                IDX_MOUTH_LEFT,
                IDX_MOUTH_RIGHT,
            ]

            points = []
            for i in indices:
                p = landmarks[i]
                points.append([p.x * W, p.y * H])

            keypoints = np.array(points, dtype=np.float32)
            # (5,2)

            # enforce left/right ordering
            if keypoints[0, 0] > keypoints[1, 0]:
                keypoints[[0, 1]] = keypoints[[1, 0]]
            if keypoints[3, 0] > keypoints[4, 0]:
                keypoints[[3, 4]] = keypoints[[4, 3]]

            # draw 5 points
            for px, py in keypoints.astype(int):
                cv2.circle(img, (int(px), int(py)), 4, (0, 255, 0), -1)

            cv2.putText(
                img,
                "5pt",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        cv2.imshow("5pt Landmarks", img)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
