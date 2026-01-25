"""
evaluate.py
Threshold tuning / evaluation using enrollment crops (aligned 112x112).
Assumptions:
- Enrollment crops exist under: data/enroll/<name>/*.jpg
- Crops are aligned (112x112) already (as saved by enroll.py / haar_5pt pipeline)
- Uses ArcFaceONNXEmbedder from embed.py (your working embedder)
Outputs:
- Prints summary stats for genuine/impostor cosine distances
- Suggests a threshold based on a target FAR
Execute:
python -m src.evaluate
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from .embed import ArcFaceONNXEmbedder


# -------------------------
# Configuration
# -------------------------
@dataclass
class EvaluationSettings:
    enroll_dir: Path = Path("data/enroll")
    min_imgs_per_person: int = 5
    max_imgs_per_person: int = 80
    target_far: float = 0.01
    thresholds: Tuple[float, float, float] = (0.10, 1.20, 0.01)
    require_size: Tuple[int, int] = (112, 112)


# -------------------------
# Math Functions
# -------------------------
def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def compute_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - compute_cosine_similarity(a, b)


# -------------------------
# IO Functions
# -------------------------
def get_person_directories(config: EvaluationSettings) -> List[Path]:
    if not config.enroll_dir.exists():
        raise FileNotFoundError(
            f"Enroll dir not found: {config.enroll_dir}. Run enroll.py first."
        )
    return sorted([p for p in config.enroll_dir.iterdir() if p.is_dir()])


def _is_aligned_crop(img: np.ndarray, req: Tuple[int, int]) -> bool:
    h, w = img.shape[:2]
    return (w, h) == (int(req[0]), int(req[1]))


def load_embeddings_for_person(
    embedding_model: ArcFaceONNXEmbedder,
    person_dir: Path,
    config: EvaluationSettings,
) -> List[np.ndarray]:
    imgs = sorted(list(person_dir.glob("*.jpg")))[: config.max_imgs_per_person]
    embs: List[np.ndarray] = []
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if config.require_size is not None and not _is_aligned_crop(img, config.require_size):
            continue
        result = embedding_model.embed(img)
        embs.append(result.embedding)
    return embs


# -------------------------
# Evaluation Functions
# -------------------------
def calculate_pairwise_distances(
    embs_a: List[np.ndarray], embs_b: List[np.ndarray], same: bool
) -> List[float]:
    distances: List[float] = []
    if same:
        for i in range(len(embs_a)):
            for j in range(i + 1, len(embs_a)):
                distances.append(compute_cosine_distance(embs_a[i], embs_a[j]))
    else:
        for ea in embs_a:
            for eb in embs_b:
                distances.append(compute_cosine_distance(ea, eb))
    return distances


def sweep_thresholds(genuine: np.ndarray, impostor: np.ndarray, config: EvaluationSettings):
    t0, t1, step = config.thresholds
    thresholds = np.arange(t0, t1 + 1e-9, step, dtype=np.float32)
    results = []
    for thr in thresholds:
        far = float(np.mean(impostor <= thr)) if impostor.size else 0.0
        frr = float(np.mean(genuine > thr)) if genuine.size else 0.0
        results.append((float(thr), far, frr))
    return results


def describe_array(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size} mean={arr.mean():.3f} std={arr.std():.3f} "
        f"p05={np.percentile(arr, 5):.3f} p50={np.percentile(arr, 50):.3f} "
        f"p95={np.percentile(arr, 95):.3f}"
    )


def main():
    config = EvaluationSettings()
    embedding_model = ArcFaceONNXEmbedder(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )
    person_directories = get_person_directories(config)
    if len(person_directories) < 1:
        print("No enrolled people found.")
        return

    embeddings_per_person: Dict[str, List[np.ndarray]] = {}
    for pdir in person_directories:
        name = pdir.name
        embs = load_embeddings_for_person(embedding_model, pdir, config)
        if len(embs) >= config.min_imgs_per_person:
            embeddings_per_person[name] = embs
        else:
            print(
                f"Skipping {name}: only {len(embs)} valid aligned crops "
                f"(need >= {config.min_imgs_per_person})."
            )

    names = sorted(embeddings_per_person.keys())
    if len(names) < 1:
        print("Not enough data to evaluate. Enroll more samples.")
        return

    # Genuine
    genuine_distances: List[float] = []
    for name in names:
        genuine_distances.extend(
            calculate_pairwise_distances(embeddings_per_person[name], embeddings_per_person[name], same=True)
        )

    # Impostor
    impostor_distances: List[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            impostor_distances.extend(
                calculate_pairwise_distances(
                    embeddings_per_person[names[i]], embeddings_per_person[names[j]], same=False
                )
            )

    genuine = np.array(genuine_distances, dtype=np.float32)
    impostor = np.array(impostor_distances, dtype=np.float32)

    print("\n=== Distance Distributions (cosine distance = 1 - cosine similarity) ===")
    print(f"Genuine (same person): {describe_array(genuine)}")
    print(f"Impostor (diff persons): {describe_array(impostor)}")

    results = sweep_thresholds(genuine, impostor, config)

    # Choose threshold with FAR <= target_far and minimal FRR
    best = None
    for thr, far, frr in results:
        if far <= config.target_far:
            if best is None or frr < best[2]:
                best = (thr, far, frr)

    print("\n=== Threshold Sweep ===")
    stride = max(1, len(results) // 10)
    for thr, far, frr in results[::stride]:
        print(f"thr={thr:.2f} FAR={far*100:5.2f}% FRR={frr*100:5.2f}%")

    if best is not None:
        thr, far, frr = best
        print(
            f"\nSuggested threshold (target FAR {config.target_far*100:.1f}%): "
            f"thr={thr:.2f} FAR={far*100:.2f}% FRR={frr*100:.2f}%"
        )
        sim_thr = 1.0 - best[0]
        print(
            f"\n(Equivalent cosine similarity threshold ~ {sim_thr:.3f}, since sim = 1 - dist)"
        )
    else:
        print(
            f"\nNo threshold in range met FAR <= {config.target_far*100:.1f}%. "
            "Try widening threshold sweep range or collecting more varied samples."
        )

    print()


if __name__ == "__main__":
    main()
