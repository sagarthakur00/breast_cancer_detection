"""
model_loader.py
Handles loading the trained DenseNet-121 model from model_best.pth.
Called once at startup by app.py — model is kept in memory for fast inference.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

# Labels matching the ImageFolder class order (alphabetical): benign=0, malignant=1
CLASSES = ["Benign", "Malignant"]

# Path to the saved checkpoint — sits at the project root
MODEL_PATH = Path(__file__).resolve().parent.parent / "model_best.pth"


def build_model() -> nn.Module:
    """
    Reconstruct the DenseNet-121 architecture with the same head used during training:
        model.classifier = nn.Linear(1024, 2)
    Weights parameter is set to None so we don't download ImageNet weights
    before overwriting them with our checkpoint.
    """
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 2)
    return model


def load_model(device: torch.device) -> nn.Module:
    """
    Load the best checkpoint saved by the trainer, move to `device`, set eval mode.
    The checkpoint format saved by the pytorch-template trainer is:
        {
            "arch": "densenet121",
            "epoch": int,
            "state_dict": OrderedDict,
            "optimizer": ...,
            ...
        }
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model_best.pth not found at {MODEL_PATH}.\n"
            "Please copy your trained model_best.pth to the project root directory."
        )

    model = build_model()

    # PyTorch ≥2.6 changed the default of weights_only to True.
    # Our checkpoint was saved by pytorch-template and includes the full
    # ConfigParser object, so we must use weights_only=False.
    # This is safe because the checkpoint was created by our own training run.
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # Support both raw state_dict and wrapped checkpoint dict
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
