# Facial Identification Using ArcFace ONNX with 5-Point Landmark Alignment

An advanced facial recognition solution designed for CPU-based systems, featuring ArcFace embeddings combined with precise 5-point facial landmark alignment. Developed with emphasis on simplicity, reliability, and deployment on standard laptops lacking GPU support.

**Developed by:** Abayo Hirwa Jovin

---

## 📋 Overview

1. [Capabilities](#capabilities)
2. [Hardware and Software Prerequisites](#hardware-and-software-prerequisites)
3. [Setup Instructions](#setup-instructions)
4. [Directory Layout](#directory-layout)
5. [Getting Started Quickly](#getting-started-quickly)
6. [Comprehensive User Manual](#comprehensive-user-manual)
7. [System Workflow](#system-workflow)
8. [Common Issues and Solutions](#common-issues-and-solutions)

---

## ✨ Capabilities

- ✅ **CPU-Based Processing**: Operates smoothly on standard computers without requiring GPU
- ✅ **5-Point Landmark Alignment**: Combines Haar cascade detection with MediaPipe landmark identification
- ✅ **ArcFace ONNX Implementation**: Generates 512-dimensional normalized embeddings
- ✅ **Component-Based Architecture**: Individual modules can be tested separately
- ✅ **Live Identification**: Supports multiple face detection with configurable matching thresholds
- ✅ **Threshold Optimization**: Statistical analysis of false acceptance and rejection rates
- ✅ **Open-Set Identification**: Automatically rejects unrecognized individuals
- ✅ **Persistent Storage**: JSON metadata combined with NPZ embedding files

---

## 🖥️ Hardware and Software Prerequisites

- **Python Version**: 3.9 or higher (validated with Python 3.11)
- **Operating Systems**: Compatible with macOS, Linux, and Windows
- **Camera Device**: Webcam necessary for live input
- **Memory**: Minimum 2GB RAM recommended
- **Disk Space**: Approximately 500MB for ONNX model and libraries

### Checking Python Version

```bash
python3 --version  # Expected output: Python 3.9 or later
```

---

## 📦 Setup Instructions

### Step 1: Download the Repository

```bash
cd /path/to/your/workspace
git clone https://github.com/AbayoHJovin
cd Face-Recog-onnx
```

### Step 2: Set Up Virtual Environment

```bash
python3.11 -m venv .venv
```

### Step 3: Enable Virtual Environment

**For macOS/Linux:**

```bash
source .venv/bin/activate
```

**For Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Update pip

```bash
python -m pip install --upgrade pip
```

### Step 5: Install Required Packages

From requirements file:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install opencv-python numpy onnxruntime scipy tqdm mediapipe protobuf
```

### Step 6: Obtain ArcFace ONNX Model

```bash
# Execute from project root
curl -L -o buffalo_l.zip "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"

unzip -o buffalo_l.zip

cp w600k_r50.onnx models/embedder_arcface.onnx

# Remove temporary files
rm -f buffalo_l.zip w600k_r50.onnx 1k3d68.onnx 2d106det.onnx det_10g.onnx genderage.onnx
```

✅ **Setup finished successfully!**

---

## 📁 Directory Layout

```
Face-Recog-onnx/
├── src/
│   ├── camera.py           # Webcam access verification
│   ├── detect.py           # Haar-based face detection validation
│   ├── landmarks.py        # MediaPipe 5-point landmark detection
│   ├── align.py            # Facial alignment to 112×112 pixels
│   ├── embed.py            # ArcFace ONNX embedding generation
│   ├── enroll.py           # Database creation utility
│   ├── evaluate.py         # Threshold analysis (FAR/FRR metrics)
│   ├── recognize.py        # Live identification system
│   └── haar_5pt.py         # Integrated Haar + 5pt detection module
├── data/
│   ├── enroll/             # Aligned 112×112 enrollment images (per individual)
│   │   ├── Person1/
│   │   ├── Person2/
│   │   └── ...
│   └── db/                 # Identification database
│       ├── face_db.npz     # Embeddings (identity → vector)
│       └── face_db.json    # Supplementary information
├── models/
│   └── embedder_arcface.onnx   # ArcFace neural network (w600k_r50)
├── book/                   # Documentation and references
├── requirements.txt        # Python package list
├── README.md              # Current documentation
└── init_project.py        # Initialization utility
```

---

## 🚀 Getting Started Quickly (5 Minutes)

### Validate Each Component

```bash
# 1. Confirm camera functionality
python -m src.camera

# 2. Verify face detection
python -m src.detect

# 3. Check 5-point landmark detection
python -m src.landmarks

# 4. Test facial alignment
python -m src.align

# 5. Validate embedding generation (model required)
python -m src.embed
```

### Register and Identify

```bash
# 6. Register individuals (repeat for multiple people)
python -m src.enroll

# 7. Analyze optimal threshold
python -m src.evaluate

# 8. Launch live identification
python -m src.recognize
```

---

## 📖 Comprehensive User Manual

### 1️⃣ Webcam Verification

**Objective:** Ensure camera access and performance

```bash
python -m src.camera
```

**Expected Behavior:**

- Camera window appears
- Motion appears fluid
- Frame rate shown
- Use **q** to close

**Problem Resolution:**

- If camera fails, modify index in code (0→1 or 0→2)
- Check permissions (macOS: Settings → Privacy & Security → Camera)

---

### 2️⃣ Face Detection Validation

**Objective:** Confirm Haar cascade face detection

```bash
python -m src.detect
```

**Expected Behavior:**

- Face enclosed in green rectangle
- Rectangle tracks face movement
- Use **q** to close

---

### 3️⃣ Landmark Detection

**Objective:** Verify 5-point facial landmark extraction

```bash
python -m src.landmarks
```

**Expected Behavior:**

- Green rectangle around face
- 5 green dots: left eye, right eye, nose, left mouth corner, right mouth corner
- Use **q** to close

---

### 4️⃣ Facial Alignment

**Objective:** Validate 112×112 alignment transformation

```bash
python -m src.align
```

**Expected Behavior:**

- Left panel: Original face with landmark points
- Right panel: Corrected 112×112 upright face
- Use **q** to exit
- Use **s** to save aligned image for inspection

**Process Description:**

- Similarity transformation applied (rotation, scaling, translation)
- Face oriented to standard position
- Result always 112×112 pixels

---

### 5️⃣ Embedding Generation

**Objective:** Confirm ArcFace ONNX model and vector creation

```bash
python -m src.embed
```

**Expected Behavior:**

```
embedding dim: 512
norm(before L2): 21.85
cos(prev,this): 0.988
```

**Explanation:**

- `embedding dim: 512` → ResNet-50 vector size
- `norm(before L2): 21.85` → Raw vector magnitude
- `cos(prev,this): 0.988` → Cosine similarity between consecutive frames (typically ~0.99)

**Commands:**

- Use **q** to exit
- Use **p** to display embedding details

---

### 6️⃣ Database Creation

**Objective:** Construct facial recognition database

```bash
python -m src.enroll
```

**Procedure:**

1. Input individual's name (e.g., "Alice", "Bob")
2. Position face in front of camera
3. Collect samples using the controls listed below
4. Use **s** to finalize and store enrollment

**Commands:**

- **SPACE** → Manually capture single sample
- **a** → Enable automatic capture (every 0.25 seconds)
- **s** → Complete enrollment (minimum 15 samples recommended)
- **r** → Clear recent samples (preserves existing images)
- **q** → Exit without saving

**Recommended Practices:**

- Maintain consistent lighting
- Slightly vary head orientation (left/right/up/down)
- Show different facial expressions (neutral, smiling, serious)
- Collect at least 15 samples per individual
- Register at least 10 different people for effective evaluation

**Results:**

- Corrected 112×112 images stored in `data/enroll/<name>/`
- Average embedding saved to `data/db/face_db.npz`
- Additional data saved to `data/db/face_db.json`

---

### 7️⃣ Threshold Optimization

**Objective:** Determine best identification threshold through statistical analysis

```bash
python -m src.evaluate
```

**Expected Behavior:**

```
=== Distance Distributions (cosine distance = 1 - cosine similarity) ===
Genuine (same person):  n=50 mean=0.345 std=0.087 p05=0.210 p50=0.328 p95=0.502
Impostor (diff persons): n=800 mean=0.812 std=0.089 p05=0.641 p50=0.821 p95=0.951

=== Threshold Sweep ===
thr=0.10 FAR=  0.00%  FRR= 98.00%
thr=0.35 FAR=  1.25%  FRR=  8.00%
thr=0.60 FAR= 12.50%  FRR=  2.00%

Suggested threshold (target FAR 1.0%): thr=0.34 FAR=1.00% FRR=10.00%

(Equivalent cosine similarity threshold ~ 0.66, since sim = 1 - dist)
```

**Understanding Results:**

- **Genuine distances** = comparisons between same individual (should be low)
- **Impostor distances** = comparisons between different individuals (should be high)
- **FAR** (False Acceptance Rate) = incorrectly accepting unknown as known
- **FRR** (False Rejection Rate) = incorrectly rejecting known individual
- **Recommended threshold** = balances acceptance and rejection rates

**Important:** Apply the recommended threshold in `recognize.py` (modify `dist_thresh=0.34`)

---

### 8️⃣ Real-Time Identification

**Objective:** Perform live facial identification using camera feed

```bash
python -m src.recognize
```

**Expected Behavior:**

- Camera feed displays detected faces
- Each face shows identity or "Unknown"
- Similarity score and distance shown
- Frame rate in top-left corner

**Commands:**

- **q** → Exit application
- **r** → Refresh database from storage
- **+** → Increase threshold (more acceptances, more false positives)
- **-** → Decrease threshold (fewer acceptances, fewer false positives)
- **d** → Enable/disable debug information

**Functionality:**

- Green rectangle + green text = identified (approved)
- Green rectangle + red text = unidentified (denied)
- Similarity indicator shows confidence level
- Temporal filtering reduces label instability

---

## 🏗️ System Workflow

### Registration Process

```
Camera Input
    ↓
Haar Face Detection
    ↓
MediaPipe 5-Point Landmark Detection
    ↓
Face Correction (112×112)
    ↓
ArcFace ONNX Vector Generation (512-dim)
    ↓
L2 Vector Normalization
    ↓
Average Calculation (multiple samples)
    ↓
Database Storage (face_db.npz)
```

### Identification Process

```
Camera Input
    ↓
Haar Face Detection
    ↓
MediaPipe 5-Point Landmark Detection
    ↓
Face Correction (112×112)
    ↓
ArcFace ONNX Vector Generation (512-dim)
    ↓
L2 Vector Normalization
    ↓
Cosine Distance vs Database Templates
    ↓
Threshold-Based Decision
    ↓
Approval/Denial + Identity Display
```

### Core Principles

**5-Point Landmark Alignment:**

- Identifies: left eye, right eye, nose tip, left mouth corner, right mouth corner
- Implements similarity transformation (rotation, scaling, translation)
- Ensures uniform input for vector generation
- Minimizes within-class variation, enhances identification accuracy

**L2 Vector Normalization:**

- Vector divided by its L2 magnitude
- Produces unit vector (magnitude = 1.0)
- Allows cosine similarity via dot product

**Cosine Distance:**

- Distance = 1 - cosine_similarity
- Scale: 0 (identical) to 2 (opposite)
- Threshold ~0.34 corresponds to similarity ~0.66

---

## 🐛 Common Issues and Solutions

### Problem: Camera fails to open

**Fix:**

```bash
# Attempt different camera index
python -m src.camera  # Try 0, 1, 2
```

**macOS Resolution:**

- Navigate to System Settings → Privacy & Security → Camera
- Grant access to Terminal or VS Code
- Restart terminal application

**Linux Resolution:**

```bash
# Verify camera permissions
ls -la /dev/video0
# If needed, add user to video group
sudo usermod -aG video $USER
```

---

### Problem: "Module mediapipe not found"

**Fix:**

```bash
pip uninstall -y mediapipe
pip install mediapipe==0.10.32
```

---

### Problem: "Model embedder_arcface.onnx not located"

**Fix:**

```bash
# Download and extract model again
curl -L -o buffalo_l.zip "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
unzip -o buffalo_l.zip
cp w600k_r50.onnx models/embedder_arcface.onnx
rm -f buffalo_l.zip w600k_r50.onnx 1k3d68.onnx 2d106det.onnx det_10g.onnx genderage.onnx
```

---

### Problem: Identification performance is inadequate

**Solutions:**

1. Register more samples (aim for 20-30 per individual)
2. Maintain uniform lighting during registration
3. Re-optimize threshold with expanded dataset
4. Inspect alignment quality (`python -m src.align`)
5. Confirm model authenticity (verify embedding norms in `embed.py`)

---

### Problem: "Embedding magnitude is 1.0 but values appear incorrect"

**Verification:**

```bash
python -m src.embed
# Use 'p' to display embedding information
# Values should range from -1 to +1
# Magnitude should be approximately 1.0
```

---

## 📊 Database Structure

### `face_db.npz`

- NumPy compressed file containing identity-to-vector mappings
- Each vector is 512-dimensional, L2-normalized float32
- Access with: `np.load('data/db/face_db.npz', allow_pickle=True)`

### `face_db.json`

- Information file with registration details
- Includes: creation time, vector size, registered identities, sample counts

---

## 🔧 Parameter Adjustment

To modify identification settings, update `src/recognize.py`:

```python
recognizer = FaceDatabaseMatcher(
    database=database,
    dist_thresh=0.34  # ← Modify this value
)
```

Lower value = stricter (fewer approvals, fewer false positives)
Higher value = more permissive (more approvals, more false positives)

---

## 📝 Important Information

- **All vectors are L2-normalized** (magnitude = 1.0)
- **Matching uses cosine distance** (dot product of normalized vectors)
- **CPU-focused implementation** ensures consistency and broad compatibility
- **Modular architecture** enables independent component replacement
- **No GPU dependency** for deployment

---

## 📚 Source Materials

1. Deng et al. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019.
2. InsightFace Framework. https://github.com/deepinsight/insightface
3. MediaPipe Framework. https://mediapipe.dev/
4. ONNX Runtime. https://onnxruntime.ai/

---

## 📄 License

Educational use. Based on Gabriel Baziramwabo's Face Recognition course.

---

## ❓ FAQ

**Q: Can I use this for production?**
A: Yes, with proper consent and legal framework. CPU performance is ~10-20 FPS.

**Q: How many faces can I enroll?**
A: Theoretically unlimited; practically tested up to 100+ identities.

**Q: Can I add GPU acceleration?**
A: Yes, change ONNX Runtime provider from `CPUExecutionProvider` to `CUDAExecutionProvider`.

**Q: What's the recognition accuracy?**
A: ~95%+ verification accuracy at 1% FAR with well-enrolled database. Depends on enrollment quality and threshold tuning.

**Q: Can I use this on mobile?**
A: ONNX Runtime supports mobile; would require porting to appropriate framework (TFLite, Core ML, etc.).

---

**Ready to build your face recognition system?** Start with Step 1 in the [Installation](#installation) section! 🚀
# Face-Recog-onnx
