import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from src import features, config

st.set_page_config(page_title="DermAI Classification", layout="centered")

st.title("🔬 Skin Lesion Classification")
st.markdown("""
This system uses **Classical Machine Vision** techniques (Histogram Equalization, Adaptive Thresholding, Morphology) 
to classify skin lesions.
""")

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

uploaded_file = st.file_uploader("Choose a dermoscopy image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)

    with st.spinner('Extracting Handcrafted Features...'):
        feat_vector = features.extract_all_features_pipeline(image)
        feat_vector = feat_vector.reshape(1, -1)
        feat_scaled = scaler.transform(feat_vector)
        probs = model.predict_proba(feat_scaled)
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]

    with col2:
        st.subheader(f"Prediction: **{pred_label}**")
        st.metric("Confidence", f"{probs[0][pred_idx] * 100:.2f}%")

    st.subheader("Class Probabilities")
    chart_data = pd.DataFrame({"Class": classes, "Probability": probs[0] * 100})
    st.bar_chart(chart_data.set_index("Class"))

    with st.expander("ℹ️  Legend: What do these abbreviations mean?"):
        st.table(pd.DataFrame(config.LEGEND_DATA))

    # --- Explainability ---
    with st.expander("See Internal Logic (Computer Vision Pipeline Steps)", expanded=False):
        st.info("Visualizing the exact steps performed by `src.features.py`")

        # Unpack the 4 preprocessing images
        img_resized, img_gray, img_eq, img_blur = features.preprocess_image(image)
        mask_raw, mask_clean, mask_connected = features.segment_lesion(img_blur)
        mask_final, _, _, _ = features.isolate_largest_component(mask_connected)
        _, texture_vis = features.compute_texture_sobel(img_gray)

        img_lesion_only = cv2.bitwise_and(img_resized, img_resized, mask=mask_final)

        # Row 1: Preprocessing (4 Steps now)
        st.markdown("### Phase 1: Preprocessing")
        c1, c2, c3, c4 = st.columns(4)
        c1.image(img_resized, channels="BGR", caption="1. Resize")
        c2.image(img_gray, caption="2. Grayscale")
        c3.image(img_eq, caption="3. Equalized (Contrast+)")
        c4.image(img_blur, caption="4. Blur (Reduce Noise)")
        st.divider()

        # Row 2: Segmentation
        st.markdown("### Phase 2: Segmentation")
        c5, c6 = st.columns(2)
        c5.image(mask_raw, caption="5. Adaptive Threshold (From Blur)")
        c6.image(mask_clean, caption="6. Morph Opening (Clean)")
        st.divider()

        # Row 3: Connection & Selection
        c7, c8 = st.columns(2)
        c7.image(mask_connected, caption="7. Morph Dilation (Connect)")
        c8.image(mask_final, caption="8. Final Mask")
        st.divider()

        # Row 4: Analysis
        st.markdown("### Phase 3: Analysis")
        c9, c10 = st.columns(2)
        c9.image(img_lesion_only, channels="BGR", caption="9. Masked Color Source")
        c10.image(texture_vis, caption="10. Sobel Texture")

        # Histogram
        st.write("**11. Lesion Color Histogram**")
        fig, ax = plt.subplots(figsize=(6, 2))
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img_resized], [i], mask_final, [256], [0, 256])
            ax.plot(hist, color=color)
            ax.set_xlim([0, 256])
        ax.set_title("Color Frequency")
        st.pyplot(fig)