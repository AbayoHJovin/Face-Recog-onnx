"""
Embedding stage (ArcFace ONNX) using the operational pipeline:
camera
-> Haar detection
-> FaceMesh 5pt
-> align_face_5pt (112x112)
-> ArcFace embedding
-> vector visualization (educational)
Execute:
python -m src.embed
Controls:
q : exit
p : print embedding stats to terminal
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import time
import cv2
import numpy as np
import onnxruntime as ort
from .haar_5pt import Haar5ptDetector, align_face_with_5_points


# -------------------------
# Data Structures
# -------------------------
@dataclass
class EmbeddingOutput:
    embedding: np.ndarray
    norm_before: float
    dim: int
    # (D,) float32, L2-normalized


# -------------------------
# Embedder Class
# -------------------------
class ArcFaceONNXEmbedder:
    """
    ArcFace / InsightFace-style ONNX embedder.
    Input: aligned 112x112 BGR image.
    Output: L2-normalized embedding vector.
    """

    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.in_w, self.in_h = input_size
        self.debug = debug
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        if debug:
            print("[embed] model loaded")
            print(
                "[embed] input:",
                self.session.get_inputs()[0].shape,
                "print([embed] output:",
                self.session.get_outputs()[0].shape,
            )

    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[:2] != (self.in_h, self.in_w):
            aligned_bgr = cv2.resize(aligned_bgr, (self.in_w, self.in_h))
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12):
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float32), n

    def embed(self, aligned_bgr: np.ndarray) -> EmbeddingOutput:
        x = self._preprocess(aligned_bgr)
        y = self.session.run([self.output_name], {self.input_name: x})[0]
        v = y.reshape(-1).astype(np.float32)
        v_norm, n0 = self._l2_normalize(v)
        return EmbeddingOutput(v_norm, n0, v_norm.size)


# -------------------------
# Visualization Helpers
# -------------------------
def render_text_lines(img, lines, origin=(10, 30), scale=0.7, color=(0, 255, 0)):
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += int(28 * scale)


def visualize_embedding_grid(
    img: np.ndarray,
    emb: np.ndarray,
    top_left=(10, 220),
    cell_scale: int = 6,
    title: str = "embedding",
):
    """
    Visualize embedding vector as a heatmap matrix.
    """
    D = emb.size
    cols = int(np.ceil(np.sqrt(D)))
    rows = int(np.ceil(D / cols))
    mat = np.zeros((rows, cols), dtype=np.float32)
    mat.flat[:D] = emb
    norm = (mat - mat.min()) / (mat.max() - mat.min() + 1e-6)
    gray = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heat = cv2.resize(
        heat,
        (cols * cell_scale, rows * cell_scale),
        interpolation=cv2.INTER_NEAREST,
    )
    x, y = top_left
    h, w = heat.shape[:2]
    ih, iw = img.shape[:2]
    if x + w > iw or y + h > ih:
        return 0, 0
    img[y : y + h, x : x + w] = heat
    cv2.putText(
        img,
        title,
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )
    return w, h


def embedding_preview_text(emb: np.ndarray, n: int = 8) -> str:
    vals = " ".join(f"{v:+.3f}" for v in emb[:n])
    return f"vec[0:{n}]: {vals} ..."


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# -------------------------
# Demo Function
# -------------------------
def main():
    camera = cv2.VideoCapture(0)
    detector = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=False,
    )
    embedding_model = ArcFaceONNXEmbedder(
        model_path="models/embedder_arcface.onnx",
        debug=False,
    )
    previous_embedding: Optional[np.ndarray] = None
    print("Embedding Demo running. Press 'q' to quit, 'p' to print embedding.")
    t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        display_image = frame.copy()
        face_list = detector.detect(frame, max_faces=1)
        display_info = []

        if face_list:
            face_obj = face_list[0]
            cv2.rectangle(display_image, (face_obj.x1, face_obj.y1), (face_obj.x2, face_obj.y2), (0, 255, 0), 2)
            for x, y in face_obj.kps.astype(int):
                cv2.circle(display_image, (x, y), 3, (0, 255, 0), -1)

            aligned_image, _ = align_face_with_5_points(frame, face_obj.kps, out_size=(112, 112))
            result = embedding_model.embed(aligned_image)
            display_info.append(f"embedding dim: {result.dim}")
            display_info.append(f"norm(before L2): {result.norm_before:.2f}")

            if previous_embedding is not None:
                sim = compute_cosine_similarity(previous_embedding, result.embedding)
                display_info.append(f"cos(prev,this): {sim:.3f}")
            previous_embedding = result.embedding

            aligned_small = cv2.resize(aligned_image, (160, 160))
            h, w = display_image.shape[:2]
            display_image[10:170, w - 170 : w - 10] = aligned_small

            render_text_lines(display_image, display_info, origin=(10, 30))
            HEAT_X, HEAT_Y = 10, 220
            CELL_SCALE = 6

            ww, hh = visualize_embedding_grid(
                display_image,
                result.embedding,
                top_left=(HEAT_X, HEAT_Y),
                cell_scale=CELL_SCALE,
                title="embedding heatmap",
            )

            if ww > 0:
                cv2.putText(
                    display_image,
                    embedding_preview_text(result.embedding),
                    (HEAT_X, HEAT_Y + hh + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    2,
                )
        else:
            render_text_lines(display_image, ["no face"], origin=(10, 30), color=(0, 0, 255))

        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()

        cv2.putText(
            display_image,
            f"fps: {fps:.1f}",
            (10, display_image.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Face Embedding", display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p") and previous_embedding is not None:
            print("[embedding]")
            print(" dim:", previous_embedding.size)
            print(" min/max:", previous_embedding.min(), previous_embedding.max())
            print(" first10:", previous_embedding[:10])

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
