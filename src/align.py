# src/align.py
"""
Face alignment demonstration using the operational pipeline:
- Haar cascade face detection (efficient)
- MediaPipe FaceMesh for 5 key points (reliable)
- ArcFace-style 5-point alignment to 112x112 (or specified size)
This resolves the issue in haar_5pt.py where the aligned display appeared
only after the loop with outdated variables.
Execute:
python -m src.align
Controls:
q exit
s save current aligned face to data/debug_aligned/<timestamp>.jpg
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Import from existing module
from .haar_5pt import Haar5ptDetector, align_face_with_5_points


def add_text(img, text: str, xy=(10, 30), scale=0.8, thickness=2):
    cv2.putText(
        img,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def show_image_safely(win: str, img: np.ndarray):
    if img is None:
        return
    cv2.imshow(win, img)


def main(
    cam_index: int = 0,
    out_size: Tuple[int, int] = (112, 112),
    mirror: bool = True,
):
    cap = cv2.VideoCapture(cam_index)

    face_detector = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=True,
    )

    out_w, out_h = int(out_size[0]), int(out_size[1])
    blank = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Directory for saving aligned snapshots
    save_dir = Path("data/debug_aligned")
    save_dir.mkdir(parents=True, exist_ok=True)

    last_aligned = blank.copy()

    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    print("Alignment running. Press 'q' to quit, 's' to save aligned face.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        detected_faces = face_detector.detect(frame, max_faces=1)
        visualization = frame.copy()
        aligned_face = None

        if detected_faces:
            face = detected_faces[0]

            # Draw box + 5 pts
            cv2.rectangle(visualization, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)

            for x, y in face.kps.astype(int):
                cv2.circle(visualization, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Align (this is the main purpose)
            aligned_face, _M = align_face_with_5_points(frame, face.kps, out_size=out_size)

            # Retain last successful aligned (so window doesn't go blank on brief misses)
            if aligned_face is not None and aligned_face.size:
                last_aligned = aligned_face

            add_text(
                visualization,
                "OK (Haar + FaceMesh 5pt)",
                (10, 30),
                0.75,
                2,
            )
        else:
            add_text(visualization, "no face", (10, 30), 0.9, 2)

        # FPS
        fps_n += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = fps_n / dt
            fps_n = 0
            fps_t0 = time.time()

        add_text(visualization, f"FPS: {fps:.1f}", (10, 60), 0.75, 2)
        add_text(
            visualization,
            f"warp: 5pt -> {out_w}x{out_h}",
            (10, 90),
            0.75,
            2,
        )

        show_image_safely("align - camera", visualization)
        show_image_safely("align - aligned", last_aligned)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts = int(time.time() * 1000)
            out_path = save_dir / f"{ts}.jpg"
            cv2.imwrite(str(out_path), last_aligned)
            print(f"[align] saved: {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
