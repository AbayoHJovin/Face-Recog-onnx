# src/detect.py
import cv2


def main():
    classifier_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_classifier = cv2.CascadeClassifier(classifier_path)

    if face_classifier.empty():
        raise RuntimeError(f"Failed to load cascade: {classifier_path}")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")

    print("Haar face detect (minimal). Press 'q' to quit.")

    while True:
        ret, img = camera.read()
        if not ret:
            break

        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # minimal but practical defaults
        detected_faces = face_classifier.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        for x, y, w, h in detected_faces:
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

        cv2.imshow("Face Detection", img)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
