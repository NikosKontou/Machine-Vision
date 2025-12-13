import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
from src import features, config

st.set_page_config(page_title="DermAI Classification", layout="centered")

# Custom styling to widen the center container
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔬 Skin Lesion Classification")
st.markdown("""
This system uses **Classical Machine Vision** techniques (CLAHE, Otsu Thresholding, Morphology) 
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

        # Reshape for model input
        feat_vector_reshaped = feat_vector.reshape(1, -1)
        feat_scaled = scaler.transform(feat_vector_reshaped)

        # Predict
        probs = model.predict_proba(feat_scaled)
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]

    with col2:
        st.subheader(f"Prediction: **{pred_label}**")
        st.metric("Confidence", f"{probs[0][pred_idx] * 100:.2f}%")

    # --- Charts ---
    st.subheader("Class Probabilities")
    chart_data = pd.DataFrame({"Class": classes, "Probability": probs[0] * 100})
    st.bar_chart(chart_data.set_index("Class"))

    with st.expander("Abbreviation information"):
        df_legend = pd.DataFrame(config.LEGEND_DATA)
        st.dataframe(
            df_legend,
            column_config={
                "More Info": st.column_config.LinkColumn(
                    "More",
                    help="Click to visit Wikipedia page",
                    display_text="🔍"
                )
            },
            hide_index=True,
            use_container_width=True
        )

    # --- LIME EXPLANATION (Local XAI) ---
    st.divider()
    st.subheader("Explainable AI (LIME)")
    st.write("#### Why was this specific image classified as " + pred_label + "?")
    st.write(
        "The chart below shows which features contributed most to the model's decision for this specific image. Green bars support the prediction, red bars contradict it.")

    try:
        # 1. Load the training sample (needed to initialize LIME)
        train_sample_path = os.path.join(config.MODEL_DIR, 'X_train_sample.npy')
        if os.path.exists(train_sample_path):
            X_train_sample = np.load(train_sample_path)
            feature_names = features.get_feature_names()

            # 2. Initialize Explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_sample,
                feature_names=feature_names,
                class_names=classes,
                mode='classification',
                verbose=False
            )

            # 3. Explain this specific instance
            # We show the Top 10 features contributing to this specific decision
            exp = explainer.explain_instance(
                data_row=feat_scaled[0],
                predict_fn=model.predict_proba,
                num_features=10,
                top_labels=1
            )

            # 4. Plot
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
        else:
            st.warning("LIME initialization data (X_train_sample.npy) not found. Re-run training.")

    except Exception as e:
        st.error(f"Could not generate explanation: {e}")

    # --- Pipeline Visualization ---
    st.divider()
    with st.expander("See Internal Logic (Computer Vision Pipeline Steps)", expanded=True):
        st.info("Visualizing the exact steps performed by `src.features.py`")

        img_resized, img_gray, img_eq, img_blur = features.preprocess_image(image)
        mask_raw, mask_clean, mask_connected = features.segment_lesion(img_blur)
        mask_final, _, _, _ = features.isolate_largest_component(mask_connected)
        _, texture_vis = features.compute_texture_canny(img_gray, mask=mask_final)
        img_lesion_only = cv2.bitwise_and(img_resized, img_resized, mask=mask_final)

        # Row 1: Preprocessing
        st.markdown("### Phase 1: Preprocessing")
        c1, c2, c3, c4 = st.columns(4)
        c1.image(img_resized, channels="BGR", caption="1. Resize")
        c2.image(img_gray, caption="2. Grayscale")
        c3.image(img_eq, caption="3. CLAHE (Smart Contrast)")
        c4.image(img_blur, caption="4. Blur (Reduce Noise)")
        st.divider()

        # Row 2: Segmentation
        st.markdown("### Phase 2: Segmentation")
        c5, c6 = st.columns(2)
        c5.image(mask_raw, caption="5. Otsu Threshold")
        c6.image(mask_clean, caption="6. Morph Opening")
        st.divider()

        # Row 3: Connection & Selection
        c7, c8 = st.columns(2)
        c7.image(mask_connected, caption="7. Morph Dilation")
        c8.image(mask_final, caption="8. Final Mask")
        st.divider()

        # Row 4: Analysis
        st.markdown("### Phase 3: Analysis")
        c9, c10 = st.columns(2)
        c9.image(img_lesion_only, channels="BGR", caption="9. Masked Source")
        c10.image(texture_vis, caption="10. Canny Edges (Masked)")

        # Histogram
        st.write("**11. Lesion Color Histogram**")
        fig, ax = plt.subplots(figsize=(10, 3))
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img_resized], [i], mask_final, [256], [0, 256])
            ax.plot(hist, color=color)
            ax.set_xlim([0, 256])
        ax.set_title("Color Frequency")
        st.pyplot(fig)