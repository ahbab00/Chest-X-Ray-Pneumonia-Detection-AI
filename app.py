"""
app.py - Interactive Streamlit Web Application for Pneumonia Detection
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "best_densenet.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .status-pneumonia {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1.2rem;
        color: #991B1B;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-normal {
        background-color: #DCFCE7;
        border: 2px solid #22C55E;
        border-radius: 10px;
        padding: 1.2rem;
        color: #166534;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained neural network...")
def load_pneumonia_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(str(MODEL_PATH))


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# -----------------------------------------------------------------------------
# Sidebar Details
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lungs.png", width=70)
    st.title("Model Intelligence")
    st.markdown("""
    **Architecture:** DenseNet121  
    **Weights:** ImageNet + Fine-Tuned Top-100  
    **Input Shape:** 224 × 224 × 3  
    """)
    st.divider()
    st.subheader("Performance on Test Set")
    st.metric(label="ROC-AUC Score", value="0.9546")
    st.metric(label="Pneumonia Sensitivity", value="96.7%")
    st.metric(label="Overall Accuracy", value="86.1%")
    st.caption("Evaluated on 624 clinical test cases (Chest X-Ray Pneumonia Dataset).")
    st.divider()
    st.warning("⚠️ **Research Prototype**: This tool is for educational and research demonstrations only. Not cleared by the FDA or intended for clinical diagnosis.")

# -----------------------------------------------------------------------------
# Main Content
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🫁 Chest X-Ray Pneumonia Detection AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep learning diagnostic assistance using transfer-learning convolutional networks.</div>', unsafe_allow_html=True)

model = load_pneumonia_model()

if model is None:
    st.error(f"❌ Model checkpoint not found at `{MODEL_PATH}`. Please run `train.py` first to train and save the model.")
    st.stop()

# Input Mode Tabs
tab_upload, tab_samples = st.tabs(["📤 Upload Your Own X-Ray", "📂 Test Sample Chest X-Rays"])

selected_image = None
image_label = "Uploaded Image"

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (JPEG or PNG)...",
        type=["jpg", "jpeg", "png"],
        help="Upload an anterior-posterior (AP/PA) chest radiograph."
    )
    if uploaded_file is not None:
        try:
            selected_image = Image.open(uploaded_file)
            image_label = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading image: {e}")

with tab_samples:
    st.write("Pick an image directly from the verified test set:")
    
    test_normal_dir = PROJECT_ROOT / "data" / "chest_xray" / "test" / "NORMAL"
    test_pneumonia_dir = PROJECT_ROOT / "data" / "chest_xray" / "test" / "PNEUMONIA"

    col_btn1, col_btn2 = st.columns(2)
    sample_choice = None
    
    normal_samples = list(test_normal_dir.glob("*.jpeg"))[:5] if test_normal_dir.exists() else []
    pneumonia_samples = list(test_pneumonia_dir.glob("*.jpeg"))[:5] if test_pneumonia_dir.exists() else []

    with col_btn1:
        st.markdown("**Normal (Healthy) Cases**")
        for sample in normal_samples:
            if st.button(f"📄 {sample.name[:24]}", key=f"btn_norm_{sample.name}"):
                sample_choice = sample

    with col_btn2:
        st.markdown("**Pneumonia Cases**")
        for sample in pneumonia_samples:
            if st.button(f"⚠️ {sample.name[:24]}", key=f"btn_pneu_{sample.name}"):
                sample_choice = sample

    if sample_choice is not None:
        try:
            selected_image = Image.open(sample_choice)
            image_label = sample_choice.name
        except Exception as e:
            st.error(f"Error loading sample: {e}")

# -----------------------------------------------------------------------------
# Diagnostics & Results
# -----------------------------------------------------------------------------
if selected_image is not None:
    st.divider()
    col_img, col_diag = st.columns([1.1, 1.3], gap="large")

    with col_img:
        st.subheader("Chest Radiograph")
        st.image(selected_image, caption=f"Analyzed: {image_label}", use_container_width=True)

    with col_diag:
        st.subheader("Diagnostic Prediction")

        # Run inference
        tensor = preprocess_image(selected_image)
        prob = float(model.predict(tensor, verbose=0)[0][0])
        is_pneumonia = prob >= 0.5
        confidence = prob if is_pneumonia else (1.0 - prob)

        if is_pneumonia:
            st.markdown(
                f'<div class="status-pneumonia">🚨 PNEUMONIA DETECTED<br>'
                f'<span style="font-size: 1rem; font-weight: normal;">Confidence: {confidence * 100:.1f}%</span></div>',
                unsafe_allow_html=True
            )
            st.warning("**Clinical Notice:** Opacities or consolidations consistent with pneumonia detected. Immediate review by a certified radiologist is strongly recommended.")
        else:
            st.markdown(
                f'<div class="status-normal">✅ NORMAL CHEST RADIOGRAPH<br>'
                f'<span style="font-size: 1rem; font-weight: normal;">Confidence: {confidence * 100:.1f}%</span></div>',
                unsafe_allow_html=True
            )
            st.info("**Notice:** Clear lung fields identified without obvious consolidation. Note: Early or atypical pneumonia may not be visible on standard radiographs.")

        st.markdown("#### Probability Distribution")
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Normal Likelihood", f"{(1.0 - prob) * 100:.1f}%")
        col_p2.metric("Pneumonia Likelihood", f"{prob * 100:.1f}%")

        # Progress bar for pneumonia probability
        st.write("**Pneumonia Risk Index:**")
        st.progress(prob)
        st.caption(f"Raw Sigmoid Output: `{prob:.4f}` (Threshold: 0.5000)")

else:
    st.info("👆 Please upload a chest X-ray image or click one of the sample cases above to see real-time detection.")
