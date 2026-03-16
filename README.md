# 🔬 Breast Cancer Detection using DenseNet-121

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-90.88%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Dataset](https://img.shields.io/badge/Dataset-BreakHis-orange)

A deep learning system for binary classification of breast cancer histopathology images as **Benign** or **Malignant**, built using a fine-tuned DenseNet-121 pretrained on ImageNet.

<p align="center">
  <img src="images/mal_ben.png" width="700"/>
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Training Results](#-training-results)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Download the Dataset](#-download-the-dataset)
- [Training the Model](#-training-the-model)
- [Running Inference](#-running-inference)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)

---

## 🧠 Project Overview

Breast cancer is one of the most prevalent cancers worldwide. Early and accurate detection through histopathological analysis is critical. This project automates that classification process using a **transfer learning** approach:

- **Input:** Microscopic histopathology images (PNG, 700×460 px, RGB)
- **Output:** `Benign` or `Malignant` prediction with confidence score
- **Approach:** Fine-tuned DenseNet-121 (pretrained on ImageNet) on the BreakHis dataset
- **Final Validation Accuracy: 90.88%**

The full pipeline covers data loading, augmentation, model training, evaluation, and a deployable web application for end users.

---

## 🏗 Model Architecture

The model uses **DenseNet-121** — a densely connected convolutional network where each layer receives feature maps from all preceding layers, enabling strong feature reuse and gradient flow.

| Component | Details |
|---|---|
| **Base Model** | DenseNet-121 pretrained on ImageNet |
| **Feature Extractor** | Frozen (all `features` layers) |
| **Custom Classifier** | `Linear(1024 → 2)` |
| **Loss Function** | Cross Entropy |
| **Optimizer** | Adam (lr=0.001, amsgrad=True) |
| **LR Scheduler** | StepLR (step=20, gamma=0.1) |
| **Input Size** | 224×224 (resized & normalized) |
| **Output Classes** | 2 — Benign / Malignant |

```
DenseNet-121
├── features (pretrained, frozen)
│   ├── conv0, norm0, relu0, pool0
│   ├── denseblock1 → transition1
│   ├── denseblock2 → transition2
│   ├── denseblock3 → transition3
│   └── denseblock4 → norm5
└── classifier: Linear(1024 → 2)   ← fine-tuned
```

The feature extractor layers are **frozen** to preserve ImageNet knowledge. Only the final classifier layer is trained, making this an efficient transfer learning setup.

---

## 📊 Dataset

**BreakHis — Breast Cancer Histopathological Image Dataset**

| Property | Details |
|---|---|
| **Source** | [Kaggle – ambarish/breakhis](https://www.kaggle.com/datasets/ambarish/breakhis) |
| **Total Images** | 7,909 |
| **Benign** | 2,480 images |
| **Malignant** | 5,429 images |
| **Image Size** | 700×460 px (resized to 224×224 for training) |
| **Format** | PNG, 3-channel RGB, 8-bit depth |
| **Magnifications** | 40X, 100X, 200X, 400X |
| **Patients** | 82 |

**Benign tumor types:** Adenosis (A), Fibroadenoma (F), Phyllodes Tumor (PT), Tubular Adenoma (TA)

**Malignant tumor types:** Ductal Carcinoma (DC), Lobular Carcinoma (LC), Mucinous Carcinoma (MC), Papillary Carcinoma (PC)

<p align="center">
  <img src="images/benign1.png" width="200"/>
  <img src="images/malignant1.png" width="200"/>
  <br/>
  <em>Left: Benign &nbsp;&nbsp;&nbsp; Right: Malignant</em>
</p>

---

## 📈 Training Results

Training was run for **15 epochs** with a 90/10 train/validation split.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.4328 | 80.60% | 0.3085 | 87.54% |
| 4 | 0.3386 | 85.77% | 0.2596 | 90.79% |
| 8 | 0.3282 | 86.07% | 0.2886 | 88.67% |
| 12 | 0.3176 | 86.63% | 0.2652 | 89.50% |
| 14 | 0.3192 | 86.50% | **0.2514** | 89.75% |
| **15** | **0.3090** | **86.65%** | 0.2657 | **90.88%** |

> ✅ **Best model** saved at epoch 14 (lowest val_loss: `0.2514`) as `model_best.pth`

#### Training & Validation Accuracy
<img src="images/train_val_acc.png" width="700"/>

#### Training & Validation Loss
<img src="images/train_val_loss.png" width="700"/>

---

## ⚙️ Installation & Environment Setup

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/sagarthakur00/breast_cancer_detection.git
cd breast_cancer_detection
```

### 2. Create a virtual environment

```bash
# Using venv
python3 -m venv bcdp
source bcdp/bin/activate        # Mac/Linux
bcdp\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes everything needed for training, the backend API, and the frontend UI:
```
torch>=2.0.0          # Model training & inference
torchvision>=0.15.0
numpy>=1.24.0
pillow>=10.0.0
fastapi>=0.110.0      # Backend API
uvicorn[standard]     # ASGI server
python-multipart      # File upload support
pydantic>=2.0.0
streamlit>=1.32.0     # Frontend UI
requests>=2.31.0
tqdm>=4.65.0          # Training progress bars
tensorboard>=2.14.0   # Training visualization
pandas>=2.0.0
kaggle>=1.6.0         # Dataset download
```

---

## 📥 Download the Dataset

### Option A — Kaggle CLI (Recommended)

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to **Settings → API → Create New Token** — this downloads `kaggle.json`
3. Place it in the correct location:

```bash
# Mac/Linux
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Windows
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

4. Download and extract the dataset:

```bash
mkdir -p data
kaggle datasets download -d ambarish/breakhis -p data/ --unzip
```

The dataset will be extracted to:
```
data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/
├── benign/
│   └── SOB/
│       ├── adenosis/
│       ├── fibroadenoma/
│       ├── phyllodes_tumor/
│       └── tubular_adenoma/
└── malignant/
    └── SOB/
        ├── ductal_carcinoma/
        ├── lobular_carcinoma/
        ├── mucinous_carcinoma/
        └── papillary_carcinoma/
```

### Option B — Manual Download

Download from [https://www.kaggle.com/datasets/ambarish/breakhis](https://www.kaggle.com/datasets/ambarish/breakhis), extract the zip file, and place the contents in the `data/` folder.

---

## 🚀 Training the Model

Ensure your virtual environment is activated and dataset is downloaded.

```bash
python train.py --config config.json
```

### Optional arguments

```bash
python train.py --config config.json --lr 0.0001 --bs 32
```

| Argument | Description | Default |
|---|---|---|
| `--config` | Path to config JSON | `config.json` |
| `--resume` | Path to checkpoint to resume from | None |
| `--lr` | Learning rate override | 0.001 |
| `--bs` | Batch size override | 16 |

### Key config settings (`config.json`)

```json
{
  "arch":        { "type": "densenet121" },
  "optimizer":   { "type": "Adam", "args": { "lr": 0.001 } },
  "loss":        "cross_entropy",
  "trainer":     { "epochs": 15, "monitor": "min val_loss", "early_stop": 10 }
}
```

### Monitor training with TensorBoard

```bash
tensorboard --logdir saved/log/
# Open http://localhost:6006 in your browser
```

Checkpoints are saved every epoch to:
```
saved/models/BCDensenet/<run_id>/
├── model_best.pth          ← Best checkpoint (lowest val_loss)
├── checkpoint-epoch1.pth
└── ...
```

---

## 🔍 Running Inference

Evaluate the trained model on the full test set:

```bash
python test.py --config config.json \
  -r saved/models/BCDensenet/<run_id>/model_best.pth
```

This outputs **loss**, **accuracy**, and **top-k accuracy** on the test set.

---

## 🌐 Deployment

The project ships a fully working **FastAPI backend** and **Streamlit frontend** that you can run locally or deploy to the cloud. Both live inside the repo — no extra setup needed beyond `pip install -r requirements.txt`.

### Architecture

```
User's Browser
      │
      │  Upload histopathology image
      ▼
┌──────────────────────┐     POST /predict      ┌──────────────────────────────┐
│  Streamlit Frontend  │  ─────────────────────▶ │  FastAPI Backend             │
│  frontend/           │ ◀───────────────────── │  backend/app.py              │
│  streamlit_app.py    │  JSON: prediction +     │  Loads model_best.pth        │
└──────────────────────┘  confidence score       │  Runs DenseNet-121 inference │
                                                 └──────────────────────────────┘
```

---

### 📁 Project Structure

```
breast_cancer_detection/
│
├── backend/                        # FastAPI prediction server
│   ├── __init__.py
│   ├── app.py                      # API routes (/predict, /health)
│   └── model_loader.py             # Loads model_best.pth at startup
│
├── frontend/                       # Streamlit web UI
│   └── streamlit_app.py            # Upload image → call API → show result
│
├── base/                           # Abstract base classes (training)
├── data_loader/                    # BreakHis dataset loader
├── model/                          # DenseNet-121 & loss definitions
├── trainer/                        # Training loop
├── logger/                         # TensorBoard writer
├── utils/                          # Utility functions
│
├── images/                         # Sample images & training plots
├── saved/                          # Checkpoints & logs (gitignored)
├── data/                           # Dataset (gitignored)
│
├── model_best.pth                  # Trained model weights (27 MB)
├── config.json                     # Training configuration
├── train.py                        # Training entry point
├── test.py                         # Evaluation entry point
├── parse_config.py                 # Config parser
└── requirements.txt                # All dependencies
```

---

### 🖥️ Run the App Locally

#### Step 1 — Clone & install

```bash
git clone https://github.com/sagarthakur00/breast_cancer_detection.git
cd breast_cancer_detection

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

#### Step 2 — Start the backend (Terminal 1)

```bash
# From the project root
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be live at **http://localhost:8000**
Interactive docs at **http://localhost:8000/docs**

#### Step 3 — Start the frontend (Terminal 2)

```bash
# From the project root
streamlit run frontend/streamlit_app.py
```

Opens automatically at **http://localhost:8501**

#### Step 4 — Upload an image

1. Open **http://localhost:8501** in your browser
2. Click **"Browse files"** and upload any PNG/JPG histopathology image
3. The app sends the image to the backend and displays:
   - ✅ **Benign** or 🚨 **Malignant** label
   - Confidence score (e.g. `99.33%`)
   - Class probability bar chart

---

### 🔌 API Reference

**`POST /predict`**

Upload an image and receive a JSON prediction.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@your_image.png"
```

**Response:**
```json
{
  "prediction": "Malignant",
  "confidence": 0.9933,
  "confidence_pct": "99.33%",
  "all_scores": {
    "Benign": 0.0067,
    "Malignant": 0.9933
  }
}
```

**`GET /health`** — Check if the server and model are loaded

```json
{ "status": "ok", "model_loaded": true, "device": "cpu" }
```

---

### ☁️ Deploy Online (Free)

#### Option 1 — Hugging Face Spaces ⭐ Recommended

Best for sharing a demo — supports Streamlit natively, free tier available.

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to **New Space → Streamlit**
3. Upload these files to the Space:
   - `frontend/streamlit_app.py` → rename to `app.py` at root
   - `backend/model_loader.py`
   - `model_best.pth`
   - `requirements.txt`
4. In `app.py`, change the backend call to load the model directly (no separate FastAPI needed on HF Spaces):

```python
# HF Spaces: load model inline instead of calling a separate API
from backend.model_loader import load_model, CLASSES
import torch
from torchvision import transforms
from PIL import Image

model = load_model(torch.device("cpu"))
# ... run inference directly
```

5. Your app goes live at:
   `https://huggingface.co/spaces/<your-username>/<space-name>`

---

#### Option 2 — Render (Full Backend + Frontend)

Deploy the backend API as a Web Service and the frontend as a separate static/web service.

1. Push this repo to GitHub (already done ✅)
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repo

**Backend service settings:**
| Field | Value |
|---|---|
| Root Directory | *(leave blank)* |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn backend.app:app --host 0.0.0.0 --port $PORT` |

**Frontend service settings:**
| Field | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run frontend/streamlit_app.py --server.port $PORT --server.address 0.0.0.0` |
| Environment Variable | `BACKEND_URL=https://your-backend.onrender.com` |

> The `BACKEND_URL` env var is read automatically by `streamlit_app.py`.

---

#### Option 3 — Railway

1. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub**
2. Select your `breast_cancer_detection` repo
3. Add two services — one for backend, one for frontend
4. Set start commands as above
5. Set `BACKEND_URL` environment variable on the frontend service to point at the backend Railway URL
6. Railway auto-detects Python and deploys — your app is live in minutes

---

## 🤝 Acknowledgements

- Dataset: [BreakHis — Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis) by Fabio Spanhol et al.
- Model template inspired by [pytorch-template](https://github.com/victoresque/pytorch-template)
- DenseNet-121: *Densely Connected Convolutional Networks* — Huang et al., CVPR 2017

---

## 📄 License

This project is licensed under the MIT License. See the [License](License) file for details.

---

<p align="center">
  Made with ❤️ for medical AI
</p>
