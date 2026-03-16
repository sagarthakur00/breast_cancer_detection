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
pip install pandas kaggle
```

`requirements.txt` includes:
```
torch>=1.1
torchvision
numpy
tqdm
tensorboard>=1.14
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

This section explains how to turn the trained model into a **web application** that any user can access — no coding required on their end.

### Architecture Overview

```
User's Browser
     │
     │  Upload image
     ▼
┌─────────────────┐        ┌──────────────────────────┐
│   Frontend UI   │──────▶│   Backend API (FastAPI)   │
│   (Streamlit)   │◀──────│   Loads model_best.pth    │
└─────────────────┘  JSON  │   Returns prediction      │
                           └──────────────────────────┘
```

---

### 🔧 Backend — FastAPI

#### Folder structure

```
deployment/
├── backend/
│   ├── app.py              ← FastAPI app
│   ├── model_best.pth      ← Copy your trained model here
│   └── requirements.txt
└── frontend/
    ├── app.py              ← Streamlit UI
    └── requirements.txt
```

#### `deployment/backend/app.py`

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

app = FastAPI(title="Breast Cancer Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Load model
model = models.densenet121(pretrained=False)
model.classifier = nn.Linear(1024, 2)
checkpoint = torch.load("model_best.pth", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

CLASSES = ["Benign", "Malignant"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = torch.softmax(model(tensor), dim=1)
        confidence, predicted = torch.max(outputs, 1)

    return {
        "prediction": CLASSES[predicted.item()],
        "confidence": f"{confidence.item() * 100:.2f}%"
    }
```

#### `deployment/backend/requirements.txt`

```
fastapi
uvicorn
torch
torchvision
pillow
python-multipart
```

#### Start the backend server

```bash
cd deployment/backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# API docs available at: http://localhost:8000/docs
```

---

### 🎨 Frontend — Streamlit

#### `deployment/frontend/app.py`

```python
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Breast Cancer Detector", page_icon="🔬")

st.title("🔬 Breast Cancer Detection")
st.markdown("Upload a histopathology image to classify it as **Benign** or **Malignant**.")

BACKEND_URL = "http://localhost:8000/predict"

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        response = requests.post(
            BACKEND_URL,
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "image/png")}
        )

    if response.status_code == 200:
        result = response.json()
        label = result["prediction"]
        confidence = result["confidence"]

        if label == "Malignant":
            st.error(f"🚨 Prediction: **{label}**  |  Confidence: {confidence}")
        else:
            st.success(f"✅ Prediction: **{label}**  |  Confidence: {confidence}")
    else:
        st.error("Error connecting to backend.")
```

#### `deployment/frontend/requirements.txt`

```
streamlit
requests
pillow
```

#### Start the frontend UI

```bash
cd deployment/frontend
pip install -r requirements.txt
streamlit run app.py

# Opens automatically at: http://localhost:8501
```

---

### 🖥️ Running the Full App Locally

Open **two terminals**:

**Terminal 1 — Backend:**
```bash
cd deployment/backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd deployment/frontend
streamlit run app.py
```

Then open your browser at **http://localhost:8501**, upload a histopathology image, and get an instant prediction!

---

### ☁️ Deploying Online (Free Options)

#### Option 1 — Hugging Face Spaces (Easiest, Free)

Hugging Face Spaces supports Streamlit apps natively.

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Create a new **Space** → select **Streamlit**
3. Upload your `app.py`, `model_best.pth`, and `requirements.txt`
4. Your app goes live at `https://huggingface.co/spaces/<username>/<space-name>`

> **Note:** For HF Spaces, modify `app.py` to load the model directly (no separate backend needed) for simplicity.

#### Option 2 — Render (Backend + Frontend)

1. Push your `deployment/` folder to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. **Backend service:**
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. **Frontend service:** Deploy similarly with `streamlit run app.py --server.port $PORT`

#### Option 3 — Railway (One-click deploy)

1. Go to [railway.app](https://railway.app)
2. Click **New Project → Deploy from GitHub**
3. Select your repo, Railway auto-detects the service
4. Set environment variables if needed and deploy

---

## 📁 Project Structure

```
breast_cancer_detection/
│
├── base/                       # Abstract base classes
│   ├── base_data_loader.py
│   ├── base_model.py
│   └── base_trainer.py
│
├── data_loader/
│   └── data_loaders.py         # BreakHis dataset loader
│
├── model/
│   ├── model.py                # DenseNet-121 + ResNet definitions
│   ├── loss.py                 # Cross entropy loss
│   └── metric.py               # Accuracy metrics
│
├── trainer/
│   └── trainer.py              # Training loop
│
├── logger/
│   └── logger.py               # Logging + TensorBoard writer
│
├── utils/
│   └── util.py                 # Utility functions
│
├── deployment/                 # Web app (add after training)
│   ├── backend/
│   │   ├── app.py              # FastAPI prediction server
│   │   └── requirements.txt
│   └── frontend/
│       ├── app.py              # Streamlit UI
│       └── requirements.txt
│
├── images/                     # Sample images & training plots
├── saved/                      # Model checkpoints & logs (gitignored)
├── data/                       # Dataset (gitignored)
├── config.json                 # Training configuration
├── train.py                    # Training entry point
├── test.py                     # Evaluation entry point
├── parse_config.py             # Config parser
└── requirements.txt
```

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
