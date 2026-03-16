"""
app.py — FastAPI backend for Breast Cancer Detection
Exposes a single POST /predict endpoint that accepts an uploaded image
and returns a JSON prediction from the trained DenseNet-121 model.

Run:
    uvicorn backend.app:app --reload
    # or from within the backend/ folder:
    uvicorn app:app --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import io

from backend.model_loader import load_model, CLASSES


# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Inference transform (must match training transforms in data_loaders.py) ───
#    Training used: RandomResizedCrop(224) + ToTensor + Normalize(0.1307, 0.3081)
#    At inference we use a deterministic center crop instead of random crop.
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
])

# ── App state (model loaded once at startup) ──────────────────────────────────
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    print(f"[startup] Loading model on device: {DEVICE}")
    app_state["model"] = load_model(DEVICE)
    print("[startup] Model ready ✓")
    yield
    app_state.clear()
    print("[shutdown] Model released.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Breast Cancer Detection API",
    description=(
        "Upload a histopathology image (PNG/JPG) to the /predict endpoint. "
        "Returns a JSON response with prediction (Benign or Malignant) and confidence score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    confidence_pct: str
    all_scores: dict[str, float]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "Breast Cancer Detection API is running.",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    model_loaded = "model" in app_state
    return {"status": "ok", "model_loaded": model_loaded, "device": str(DEVICE)}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="Histopathology image (PNG or JPG)")):
    """
    Accept an uploaded image, run DenseNet-121 inference, return prediction.

    - **file**: PNG or JPG histopathology image
    - Returns: prediction label, confidence score (0–1), formatted percentage, and all class scores
    """

    # ── Validate content type ──────────────────────────────────────────────
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload a PNG or JPG image.",
        )

    # ── Read & decode image ────────────────────────────────────────────────
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image. Ensure the file is a valid PNG/JPG.")

    # ── Preprocess ─────────────────────────────────────────────────────────
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

    # ── Inference ──────────────────────────────────────────────────────────
    model = app_state["model"]
    with torch.no_grad():
        logits = model(tensor)                          # (1, 2)
        probs = torch.softmax(logits, dim=1)[0]        # (2,)

    predicted_idx = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_idx].item())
    label = CLASSES[predicted_idx]

    all_scores = {cls: round(float(probs[i].item()), 4) for i, cls in enumerate(CLASSES)}

    return PredictionResponse(
        prediction=label,
        confidence=round(confidence, 4),
        confidence_pct=f"{confidence * 100:.2f}%",
        all_scores=all_scores,
    )
