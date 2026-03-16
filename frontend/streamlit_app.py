"""
streamlit_app.py — Breast Cancer Detection Frontend
Upload a histopathology image and get an instant AI prediction.

Run:
    streamlit run frontend/streamlit_app.py
"""

import streamlit as st
import requests
from PIL import Image
import io
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Detection AI",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Backend URL (override via env var for deployment) ─────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        margin-top: 1rem;
    }
    .benign   { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .malignant { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .confidence-bar { margin-top: 0.5rem; }
    .disclaimer {
        font-size: 0.75rem;
        color: #888;
        border-top: 1px solid #ddd;
        padding-top: 0.8rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/microscope.png", width=72)
    st.title("About")
    st.markdown(
        """
        This tool uses a **DenseNet-121** deep learning model
        trained on the [BreakHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis)
        to classify breast tissue histopathology images as:

        - ✅ **Benign** — non-cancerous
        - 🚨 **Malignant** — cancerous

        **Model accuracy:** 90.88% (validation)

        ---
        **Dataset:** 7,909 images across 40X–400X magnification

        **Architecture:** DenseNet-121 (pretrained on ImageNet, fine-tuned)
        """
    )
    st.markdown("---")
    st.markdown("**Backend status:**")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if health.status_code == 200:
            st.success("🟢 API is online")
        else:
            st.warning("🟡 API responded with error")
    except Exception:
        st.error("🔴 Cannot reach backend\nMake sure it's running on " + BACKEND_URL)

# ── Main Header ───────────────────────────────────────────────────────────────
st.title("🔬 Breast Cancer Detection AI")
st.markdown(
    "Upload a **histopathology image** (PNG or JPG) to classify it as "
    "**Benign** or **Malignant** using a fine-tuned DenseNet-121 model."
)
st.markdown("---")

# ── File Uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Choose a histopathology image",
    type=["png", "jpg", "jpeg"],
    help="Upload a PNG or JPG breast tissue microscopy image.",
)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_column_width=True)

    with col2:
        st.markdown("#### Prediction Result")

        with st.spinner("🧠 Analyzing image..."):
            try:
                # Reset file pointer before sending
                uploaded_file.seek(0)
                response = requests.post(
                    PREDICT_ENDPOINT,
                    files={"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)},
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    label      = result["prediction"]        # "Benign" or "Malignant"
                    confidence = result["confidence"]        # 0.0 – 1.0
                    conf_pct   = result["confidence_pct"]   # "92.34%"
                    all_scores = result["all_scores"]        # {"Benign": x, "Malignant": y}

                    # ── Result box ─────────────────────────────────────────
                    css_class = "benign" if label == "Benign" else "malignant"
                    icon      = "✅" if label == "Benign" else "🚨"

                    st.markdown(
                        f'<div class="result-box {css_class}">'
                        f'{icon} {label}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Confidence score ───────────────────────────────────
                    st.markdown(f"**Confidence:** {conf_pct}")
                    st.progress(confidence)

                    # ── Class probability breakdown ────────────────────────
                    st.markdown("#### Class Probabilities")
                    for cls, score in all_scores.items():
                        bar_icon = "🟢" if cls == "Benign" else "🔴"
                        st.markdown(f"{bar_icon} **{cls}:** {score * 100:.2f}%")
                        st.progress(score)

                elif response.status_code == 415:
                    st.error("❌ Unsupported file type. Please upload a PNG or JPG image.")
                elif response.status_code == 400:
                    st.error("❌ Could not process the image. Please try a different file.")
                else:
                    st.error(f"❌ Server error ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to the backend API.\n\n"
                    f"Make sure the backend is running:\n```\nuvicorn backend.app:app --reload\n```"
                )
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The backend may be overloaded.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

    # ── Image metadata ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Image Info")
    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("File Name", uploaded_file.name)
    ic2.metric("Size", f"{image.size[0]}×{image.size[1]} px")
    ic3.metric("Format", image.format or uploaded_file.type.split("/")[-1].upper())

else:
    # ── Empty state placeholder ───────────────────────────────────────────
    st.info(
        "👆 Upload a histopathology image above to get started.\n\n"
        "The model will classify it as **Benign** or **Malignant** in seconds."
    )
    st.markdown("#### Sample images")
    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        st.image("images/benign1.png", caption="Example: Benign", use_column_width=True)
    with sample_col2:
        st.image("images/malignant1.png", caption="Example: Malignant", use_column_width=True)

# ── Footer disclaimer ─────────────────────────────────────────────────────────
st.markdown(
    '<p class="disclaimer">⚠️ This tool is for educational and research purposes only. '
    'It is not a substitute for professional medical diagnosis. '
    'Always consult a qualified medical professional for clinical decisions.</p>',
    unsafe_allow_html=True,
)
