# src/enroll.py
"""
enroll.py
Enrollment tool using the operational pipeline:
camera -> Haar detection -> FaceMesh 5pt -> align_face_5pt (112x112) -> ArcFace embedding
Stores template per identity (mean embedding, L2-normalized).
Re-enroll behavior:
- If data/enroll/<name> already contains aligned crops, those are loaded,
embedded again, and INCLUDED in the template. New captures are appended.
Outputs:
- data/db/face_db.npz
(name -> embedding vector)
- data/db/face_db.json (metadata)
Optional:
- data/enroll/<name>/*.jpg aligned face crops
Controls:
- SPACE: capture one sample (if face found)
- a: auto-capture toggle (captures periodically)
- s: save enrollment (after enough total samples)
- r: reset NEW samples (keeps existing crops on disk)
- q: quit
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from .haar_5pt import Haar5ptDetector, align_face_with_5_points
from .embed import ArcFaceONNXEmbedder


# -------------------------
# Configuration
# -------------------------
@dataclass
class EnrollmentSettings:
    out_db_npz: Path = Path("data/db/face_db.npz")
    out_db_json: Path = Path("data/db/face_db.json")
    save_crops: bool = True
    crops_dir: Path = Path("data/enroll")
    samples_needed: int = 15
    auto_capture_every_s: float = 0.25
    max_existing_crops: int = 300
    # UI
    window_main: str = "enroll"
    window_aligned: str = "aligned_112"


# -------------------------
# Database Helpers
# -------------------------
def create_required_directories(config: EnrollmentSettings) -> None:
    config.out_db_npz.parent.mkdir(parents=True, exist_ok=True)
    config.out_db_json.parent.mkdir(parents=True, exist_ok=True)
    if config.save_crops:
        config.crops_dir.mkdir(parents=True, exist_ok=True)


def load_database(config: EnrollmentSettings) -> Dict[str, np.ndarray]:
    if config.out_db_npz.exists():
        data = np.load(config.out_db_npz, allow_pickle=True)
        return {k: data[k].astype(np.float32) for k in data.files}
    return {}


def save_database(config: EnrollmentSettings, database: Dict[str, np.ndarray], metadata: dict) -> None:
    create_required_directories(config)
    np.savez(config.out_db_npz, **{k: v.astype(np.float32) for k, v in database.items()})
    config.out_db_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def compute_mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """Mean + L2 normalize."""
    E = np.stack([e.reshape(-1) for e in embeddings], axis=0).astype(np.float32)
    m = E.mean(axis=0)
    m = m / (np.linalg.norm(m) + 1e-12)
    return m.astype(np.float32)


# -------------------------
# Crops Loader
# -------------------------
def _get_existing_crop_files(person_folder: Path, max_count: int) -> List[Path]:
    if not person_folder.exists():
        return []
    files = sorted([p for p in person_folder.glob("*.jpg") if p.is_file()])
    if len(files) > max_count:
        files = files[-max_count:]
    return files


def load_previous_samples_from_disk(
    config: EnrollmentSettings,
    embedder: ArcFaceONNXEmbedder,
    person_folder: Path,
) -> List[np.ndarray]:
    """
    Reads aligned crops from disk and re-embeds them.
    """
    if not config.save_crops:
        return []
    crops = _get_existing_crop_files(person_folder, config.max_existing_crops)
    base: List[np.ndarray] = []
    for p in crops:
        img = cv2.imread(str(p))
        if img is None:
            continue
        try:
            result = embedder.embed(img)
            base.append(result.embedding)
        except Exception:
            continue
    return base


# -------------------------
# UI Helpers
# -------------------------
def display_enrollment_status(
    frame: np.ndarray,
    person_name: str,
    existing_count: int,
    new_count: int,
    required: int,
    auto_capture: bool,
    message: str = "",
) -> None:
    total_count = existing_count + new_count
    lines = [
        f"ENROLL: {person_name}",
        f"Existing: {existing_count} | New: {new_count} | Total: {total_count} / {required}",
        f"Auto: {'ON' if auto_capture else 'OFF'} (toggle: a)",
        "SPACE=capture | s=save | r=reset NEW | q=quit",
    ]
    if message:
        lines.insert(0, message)

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 26


# -------------------------
# Main Function
# -------------------------
def main():
    config = EnrollmentSettings()
    create_required_directories(config)

    person_name = input("Enter person name to enroll (e.g., Alice): ").strip()
    if not person_name:
        print("No name provided. Exiting.")
        return

    face_detector = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)
    embedder = ArcFaceONNXEmbedder(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )

    database = load_database(config)
    person_folder = config.crops_dir / person_name
    if config.save_crops:
        person_folder.mkdir(parents=True, exist_ok=True)

    existing_embeddings: List[np.ndarray] = load_previous_samples_from_disk(
        config, embedder, person_folder
    )
    new_embeddings: List[np.ndarray] = []

    status_message = ""
    if existing_embeddings:
        status_message = f"Loaded {len(existing_embeddings)} existing samples from disk."

    auto_capture = False
    last_auto_capture_time = 0.0

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Failed to open camera.")

    cv2.namedWindow(config.window_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(config.window_aligned, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.window_aligned, 240, 240)

    print("\nEnrollment started.")
    if existing_embeddings:
        print(
            f"Re-enroll mode: found {len(existing_embeddings)} existing samples in {person_folder}/"
        )

    print("Tip: stable lighting, move slightly left/right, different expressions.")
    print("Controls: SPACE=capture, a=auto, s=save, r=reset NEW, q=quit\n")

    t0 = time.time()
    frames = 0
    fps: Optional[float] = None

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            display_frame = frame.copy()
            detected_faces = face_detector.detect(frame, max_faces=1)
            aligned_face: Optional[np.ndarray] = None

            if detected_faces:
                face = detected_faces[0]
                cv2.rectangle(display_frame, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                for x, y in face.kps.astype(int):
                    cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                aligned_face, _ = align_face_with_5_points(frame, face.kps, out_size=(112, 112))
                cv2.imshow(config.window_aligned, aligned_face)
            else:
                cv2.imshow(config.window_aligned, np.zeros((112, 112, 3), dtype=np.uint8))

            now = time.time()
            if (
                auto_capture
                and aligned_face is not None
                and (now - last_auto_capture_time) >= config.auto_capture_every_s
            ):
                embedding_result = embedder.embed(aligned_face)
                new_embeddings.append(embedding_result.embedding)
                last_auto_capture_time = now
                status_message = f"Auto captured NEW ({len(new_embeddings)})"
                if config.save_crops:
                    filename = person_folder / f"{int(now * 1000)}.jpg"
                    cv2.imwrite(str(filename), aligned_face)

            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            if fps is not None:
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (10, display_frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            display_enrollment_status(
                display_frame,
                person_name=person_name,
                existing_count=len(existing_embeddings),
                new_count=len(new_embeddings),
                required=config.samples_needed,
                auto_capture=auto_capture,
                message=status_message,
            )

            cv2.imshow(config.window_main, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("a"):
                auto_capture = not auto_capture
                status_message = f"Auto mode {'ON' if auto_capture else 'OFF'}"
            if key == ord("r"):
                new_embeddings.clear()
                status_message = "NEW samples reset (existing kept)."
            if key == ord(" "):
                if aligned_face is None:
                    status_message = "No face detected. Not captured."
                else:
                    embedding_result = embedder.embed(aligned_face)
                    new_embeddings.append(embedding_result.embedding)
                    status_message = f"Captured NEW ({len(new_embeddings)})"
                    if config.save_crops:
                        filename = person_folder / f"{int(time.time() * 1000)}.jpg"
                        cv2.imwrite(str(filename), aligned_face)
            if key == ord("s"):
                total_samples = len(existing_embeddings) + len(new_embeddings)
                if total_samples < max(3, config.samples_needed // 2):
                    status_message = f"Not enough total samples to save (have {total_samples})."
                    continue

                combined_samples = existing_embeddings + new_embeddings
                average_embedding = compute_mean_embedding(combined_samples)
                database[person_name] = average_embedding

                metadata = {
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "embedding_dim": int(average_embedding.size),
                    "names": sorted(database.keys()),
                    "samples_existing_used": int(len(existing_embeddings)),
                    "samples_new_used": int(len(new_embeddings)),
                    "samples_total_used": int(len(combined_samples)),
                    "note": "Embeddings are L2-normalized vectors. Matching uses cosine similarity.",
                }

                save_database(config, database, metadata)
                status_message = f"Saved '{person_name}' to DB. Total identities: {len(database)}"
                print(status_message)

                existing_embeddings = load_previous_samples_from_disk(config, embedder, person_folder)
                new_embeddings.clear()

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
