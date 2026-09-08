import os
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "best_densenet.keras"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .status-box-pos {
        background-color: #fee2e2;
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
        color: #991b1b;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }
    .status-box-neg {
        background-color: #dcfce7;
        border: 2px solid #22c55e;
        border-radius: 8px;
        padding: 1rem;
        color: #166534;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(str(MODEL_PATH))


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


with st.sidebar:
    st.title("Model Details")
    st.write("**Model:** DenseNet121 (Fine-tuned)")
    st.write("**Input Size:** 224x224")
    st.divider()
    st.subheader("Test Metrics")
    st.metric("ROC-AUC", "0.9546")
    st.metric("Sensitivity", "96.7%")
    st.metric("Accuracy", "86.1%")
    st.divider()
    st.caption("Note: Prototype model for demonstration and educational testing.")

st.markdown('<div class="main-title">Chest X-Ray Pneumonia Classifier</div>', unsafe_allow_html=True)
st.write("Upload a chest X-ray image or pick from test samples to classify.")

model = get_model()
if model is None:
    st.error("Model file not found. Train the model first.")
    st.stop()

tab_upload, tab_samples = st.tabs(["Upload Image", "Sample Cases"])
selected_image = None
image_label = ""

with tab_upload:
    uploaded_file = st.file_uploader("Select X-Ray Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        selected_image = Image.open(uploaded_file)
        image_label = uploaded_file.name

with tab_samples:
    samples_normal = PROJECT_ROOT / "samples" / "NORMAL"
    samples_pneumonia = PROJECT_ROOT / "samples" / "PNEUMONIA"
    data_normal = PROJECT_ROOT / "data" / "chest_xray" / "test" / "NORMAL"
    data_pneumonia = PROJECT_ROOT / "data" / "chest_xray" / "test" / "PNEUMONIA"

    norm_dir = samples_normal if samples_normal.exists() and any(samples_normal.iterdir()) else data_normal
    pneu_dir = samples_pneumonia if samples_pneumonia.exists() and any(samples_pneumonia.iterdir()) else data_pneumonia

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Normal Samples**")
        for f in (list(norm_dir.glob("*.jpeg")) + list(norm_dir.glob("*.jpg")))[:4]:
            if st.button(f.name[:20], key=f"n_{f.name}"):
                selected_image = Image.open(f)
                image_label = f.name

    with c2:
        st.write("**Pneumonia Samples**")
        for f in (list(pneu_dir.glob("*.jpeg")) + list(pneu_dir.glob("*.jpg")))[:4]:
            if st.button(f.name[:20], key=f"p_{f.name}"):
                selected_image = Image.open(f)
                image_label = f.name

if selected_image:
    st.divider()
    col1, col2 = st.columns([1, 1.2], gap="medium")

    with col1:
        st.image(selected_image, caption=image_label, use_container_width=True)

    with col2:
        tensor = preprocess_image(selected_image)
        prob = float(model.predict(tensor, verbose=0)[0][0])
        is_pneumonia = prob >= 0.5
        conf = prob if is_pneumonia else (1.0 - prob)

        if is_pneumonia:
            st.markdown(
                f'<div class="status-box-pos">PNEUMONIA DETECTED<br>'
                f'<span style="font-size: 0.95rem; font-weight: normal;">Confidence: {conf * 100:.1f}%</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="status-box-neg">NORMAL (NO PNEUMONIA)<br>'
                f'<span style="font-size: 0.95rem; font-weight: normal;">Confidence: {conf * 100:.1f}%</span></div>',
                unsafe_allow_html=True
            )

        st.write("")
        st.write("#### Probabilities")
        m1, m2 = st.columns(2)
        m1.metric("Normal", f"{(1.0 - prob) * 100:.1f}%")
        m2.metric("Pneumonia", f"{prob * 100:.1f}%")

        st.progress(prob)
        st.caption(f"Output Score: {prob:.4f}")
else:
    st.info("Upload an image above or choose a sample to run inference.")

