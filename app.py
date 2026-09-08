import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import streamlit as st
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "best_densenet.keras"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, polished look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(30, 58, 138, 0.15);
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #dbeafe;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }
    
    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    
    .status-badge-pneumonia {
        background: #fef2f2;
        border: 1.5px solid #ef4444;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.12);
    }
    .status-badge-pneumonia h2 {
        color: #b91c1c;
        margin: 0;
        font-size: 1.35rem;
        font-weight: 700;
    }
    .status-badge-pneumonia p {
        color: #7f1d1d;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    
    .status-badge-normal {
        background: #f0fdf4;
        border: 1.5px solid #22c55e;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.12);
    }
    .status-badge-normal h2 {
        color: #15803d;
        margin: 0;
        font-size: 1.35rem;
        font-weight: 700;
    }
    .status-badge-normal p {
        color: #14532d;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    
    .prob-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #4b5563;
        margin-bottom: 0.2rem;
    }
    
    .risk-high {
        color: #dc2626;
        font-weight: 700;
    }
    .risk-low {
        color: #16a34a;
        font-weight: 700;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    .sample-pill {
        display: inline-block;
        background: #f3f4f6;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        color: #374151;
        margin-bottom: 0.5rem;
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


# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Diagnostics Engine")
    st.write("**Model:** DenseNet121 Transfer Learning")
    st.write("**Image Resolution:** 224 × 224")
    
    st.markdown("---")
    st.markdown("### 📊 Verified Performance")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("ROC-AUC", "0.955")
        st.metric("Accuracy", "86.1%")
    with col_m2:
        st.metric("Sensitivity", "96.7%")
        st.metric("Specificity", "68.4%")
    st.caption("Benchmark on 624 clinical test cases.")
    
    st.markdown("---")
    st.markdown("### 🔍 Image Enhancement")
    st.caption("Adjust view parameters for radiograph inspection:")
    enhance_contrast = st.slider("Contrast", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    enhance_brightness = st.slider("Brightness", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    invert_colors = st.checkbox("Invert Colors (Negative)", value=False)
    
    st.markdown("---")
    st.caption("⚠️ Research & educational prototype. Not a certified medical device.")

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🫁 Chest Radiograph Pneumonia Detection</h1>
    <p class="header-sub">Deep Learning diagnostic support using convolutional transfer learning</p>
</div>
""", unsafe_allow_html=True)

model = get_model()
if model is None:
    st.error("Model weights file not found. Please train the model first.")
    st.stop()

# Selection tabs
tab_upload, tab_samples = st.tabs(["📤 Upload Custom X-Ray", "📂 Verified Test Samples"])

selected_image = None
image_label = ""

with tab_upload:
    st.write("")
    uploaded_file = st.file_uploader(
        "Upload a chest radiograph (AP or PA projection):",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPEG, PNG"
    )
    if uploaded_file:
        try:
            selected_image = Image.open(uploaded_file)
            image_label = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab_samples:
    st.write("")
    samples_normal = PROJECT_ROOT / "samples" / "NORMAL"
    samples_pneumonia = PROJECT_ROOT / "samples" / "PNEUMONIA"
    data_normal = PROJECT_ROOT / "data" / "chest_xray" / "test" / "NORMAL"
    data_pneumonia = PROJECT_ROOT / "data" / "chest_xray" / "test" / "PNEUMONIA"

    norm_dir = samples_normal if samples_normal.exists() and any(samples_normal.iterdir()) else data_normal
    pneu_dir = samples_pneumonia if samples_pneumonia.exists() and any(samples_pneumonia.iterdir()) else data_pneumonia

    col_s1, col_s2 = st.columns(2)
    
    norm_files = (list(norm_dir.glob("*.jpeg")) + list(norm_dir.glob("*.jpg")))[:3] if norm_dir.exists() else []
    pneu_files = (list(pneu_dir.glob("*.jpeg")) + list(pneu_dir.glob("*.jpg")))[:3] if pneu_dir.exists() else []

    with col_s1:
        st.markdown("**🟢 Healthy Controls (Normal)**")
        for idx, f in enumerate(norm_files, 1):
            col_th, col_btn = st.columns([1, 2.5])
            with col_th:
                try:
                    thumb = Image.open(f)
                    st.image(thumb, width=64)
                except Exception:
                    pass
            with col_btn:
                if st.button(f"Analyze Normal #{idx}", key=f"norm_btn_{f.name}", use_container_width=True):
                    selected_image = Image.open(f)
                    image_label = f"Sample Normal Case #{idx}"

    with col_s2:
        st.markdown("**🔴 Confirmed Pneumonia Cases**")
        for idx, f in enumerate(pneu_files, 1):
            col_th, col_btn = st.columns([1, 2.5])
            with col_th:
                try:
                    thumb = Image.open(f)
                    st.image(thumb, width=64)
                except Exception:
                    pass
            with col_btn:
                if st.button(f"Analyze Pneumonia #{idx}", key=f"pneu_btn_{f.name}", use_container_width=True):
                    selected_image = Image.open(f)
                    image_label = f"Sample Pneumonia Case #{idx}"

# Analysis Display
if selected_image is not None:
    st.markdown("---")
    
    # Apply optional image adjustments for visual inspection
    display_img = selected_image.convert("RGB")
    if invert_colors:
        display_img = ImageOps.invert(display_img)
    if enhance_contrast != 1.0:
        display_img = ImageEnhance.Contrast(display_img).enhance(enhance_contrast)
    if enhance_brightness != 1.0:
        display_img = ImageEnhance.Brightness(display_img).enhance(enhance_brightness)

    col_view, col_diag = st.columns([1.1, 1.4], gap="large")

    with col_view:
        st.subheader("🖼️ Radiograph View")
        st.image(display_img, caption=image_label, use_container_width=True)
        w, h = selected_image.size
        st.caption(f"Original Resolution: {w} × {h} px | Processed Resolution: 224 × 224 px")

    with col_diag:
        st.subheader("📋 AI Diagnostic Summary")
        
        # Inference
        tensor = preprocess_image(selected_image)
        prob = float(model.predict(tensor, verbose=0)[0][0])
        is_pneumonia = prob >= 0.5
        confidence = prob if is_pneumonia else (1.0 - prob)
        
        # Diagnostic Card
        if is_pneumonia:
            st.markdown(f"""
            <div class="status-badge-pneumonia">
                <h2>🚨 PNEUMONIA DETECTED</h2>
                <p>Confidence Level: <strong>{confidence * 100:.1f}%</strong> (High Risk)</p>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.warning("Pulmonary opacities or infiltrates detected. Clinical evaluation and radiologist review recommended.")
        else:
            st.markdown(f"""
            <div class="status-badge-normal">
                <h2>✅ NORMAL RADIOGRAPH</h2>
                <p>Confidence Level: <strong>{confidence * 100:.1f}%</strong> (Low Risk)</p>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.success("Clear lung fields without evident focal consolidation.")

        st.markdown("#### Probability Distribution")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Normal Probability", f"{(1.0 - prob) * 100:.1f}%")
        with m_col2:
            st.metric("Pneumonia Probability", f"{prob * 100:.1f}%")

        # Custom progress meter
        st.markdown('<p class="prob-label">Pneumonia Index Gauge (0.0 = Clear, 1.0 = Consolidation)</p>', unsafe_allow_html=True)
        st.progress(prob)
        
        st.markdown(f"**Sigmoid Score:** `{prob:.4f}` *(Threshold: 0.5000)*")

else:
    st.write("")
    st.info("💡 **Getting started:** Upload a chest X-ray image above, or click any of the sample cases to evaluate.")
