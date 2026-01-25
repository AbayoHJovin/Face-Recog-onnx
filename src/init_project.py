from pathlib import Path

# Canonical project structure
project_structure = {
    "data/enroll": [],
    "data/db": [],
    "models": [
        "embedder_arcface.onnx",
    ],
    "src": [
        "camera.py",
        "detect.py",
        "landmarks.py",
        "align.py",
        "embed.py",
        "enroll.py",
        "recognize.py",
        "evaluate.py",
        "haar_5pt.py",
    ],
    "book": [],
}

for directory, file_list in project_structure.items():
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)

    for file in file_list:
        file_path = directory_path / file
        if not file_path.exists():
            file_path.touch()

print("face-recognition-5pt project structure created successfully.")
