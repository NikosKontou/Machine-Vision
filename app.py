import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os
from src import features, config

# --- Setup ---
st.set_page_config(page_title="DermAI Classification", layout="centered")

st.title("🔬 Skin Lesion Classification")
st.markdown("""
This system uses **Classical Machine Vision** techniques (Adaptive Thresholding, Morphology, Sobel Edges) 
to classify skin lesions.
""")

# --- Load Model ---
try:
    model_path = os.path.join(config.MODEL_DIR, 'skin_cancer_model.pkl')
    scaler_path = os.path.join(config.MODEL_DIR, 'scaler.pkl')
    classes_path = os.path.join(config.MODEL_DIR, 'classes.pkl')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    classes = joblib.load(classes_path)
    st.success("System Ready: Model Loaded Successfully")
except FileNotFoundError:
    st.error("Model files not found. Please run 'train_main.py' first.")
    st.stop()

# --- UI Logic ---
uploaded_file = st.file_uploader("Choose a dermoscopy image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)

    # --- Prediction ---
    with st.spinner('Extracting Handcrafted Features...'):
        # 1. extract
        feat_vector = features.extract_all_features_pipeline(image)
        # 2. reshape & scale
        feat_vector = feat_vector.reshape(1, -1)
        feat_scaled = scaler.transform(feat_vector)
        # 3. predict
        probs = model.predict_proba(feat_scaled)
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]

    with col2:
        st.subheader(f"Prediction: **{pred_label}**")
        st.metric("Confidence", f"{probs[0][pred_idx] * 100:.2f}%")

    # --- Charts & Legend ---
    st.subheader("Class Probabilities")
    chart_data = pd.DataFrame({"Class": classes, "Probability": probs[0] * 100})
    st.bar_chart(chart_data.set_index("Class"))

    with st.expander("ℹ️  Legend: What do these abbreviations mean?"):
        st.table(pd.DataFrame(config.LEGEND_DATA))

    # --- Explainability (Visualizing the Pipeline) ---
    with st.expander("See Internal Logic (Computer Vision Pipeline Steps)", expanded=False):
        st.info("Visualizing the exact steps performed by `src.features.py`")

        # We call the individual functions from our package to get the images
        img_resized, img_gray, img_blur = features.preprocess_image(image)
        mask_raw, mask_clean, mask_connected = features.segment_lesion(img_blur)
        mask_final, _, _, _ = features.isolate_largest_component(mask_connected)
        _, texture_vis = features.compute_texture_sobel(img_gray)

        # Row 1: Preprocessing
        c1, c2, c3 = st.columns(3)
        c1.image(img_resized, channels="BGR", caption="1. Resize")
        c2.image(img_gray, caption="2. Grayscale")
        c3.image(img_blur, caption="3. Gaussian Blur")
        st.divider()

        # Row 2: Segmentation
        c4, c5 = st.columns(2)
        c4.image(mask_raw, caption="4. Adaptive Threshold (Raw)")
        c5.image(mask_clean, caption="5. Morph Opening (Clean Noise)")
        st.divider()

        # Row 3: Connection & Selection
        c6, c7 = st.columns(2)
        c6.image(mask_connected, caption="6. Morph Dilation (Connect)")
        c7.image(mask_final, caption="7. Largest Component Only")
        st.divider()

        # Row 4: Texture
        st.image(texture_vis, caption="8. Sobel Texture Magnitude")