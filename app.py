import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import skew


# --- Re-implement feature extraction to ensure consistency ---
def extract_features_deployment(image_array):
    IMG_SIZE = 224
    img = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))

    features = []

    # Color Stats
    for i in range(3):
        channel = img[:, :, i]
        features.append(np.mean(channel))
        features.append(np.std(channel))
        features.append(skew(channel.flatten()))

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > 0:
            compactness = (4 * np.pi * area) / (perimeter ** 2)
        else:
            compactness = 0
    else:
        area, perimeter, compactness = 0, 0, 0

    features.extend([area, perimeter, compactness])

    # Texture (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    features.append(np.mean(mag))

    return np.array(features).reshape(1, -1)


# --- App Layout ---
st.set_page_config(page_title="DermAI Classification", layout="centered")

st.title("🔬 Skin Lesion Classification")
st.markdown("""
This system uses **Classical Machine Vision** techniques (Otsu Thresholding, Sobel Edges, 
Color Moments) rather than Deep Learning to classify skin lesions.
""")

# Load Model
try:
    model = joblib.load('models/skin_cancer_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    classes = joblib.load('models/classes.pkl')
    st.success("System Ready: Model Loaded Successfully")
except FileNotFoundError:
    st.error("Model files not found. Please run 'train_model.py' first.")
    st.stop()

# File Upload
uploaded_file = st.file_uploader("Choose a dermoscopy image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)

    # Process
    with st.spinner('Extracting Handcrafted Features...'):
        # Extract
        features = extract_features_deployment(image)
        # Scale
        features_scaled = scaler.transform(features)
        # Predict
        probs = model.predict_proba(features_scaled)
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]

    # Results
    with col2:
        st.subheader(f"Prediction: **{pred_label}**")
        # FIXED: probs is 2D array [[p1, p2...]], take probs[0]
        st.metric("Confidence", f"{probs[0][pred_idx] * 100:.2f}%")

    # Detailed Chart
    st.subheader("Class Probabilities")
    # FIXED: Ensure 1D array for Chart
    chart_data = pd.DataFrame({
        "Class": classes,
        "Probability": probs[0] * 100
    })
    st.bar_chart(chart_data.set_index("Class"))

    # Feature Visualization (Explainability)
    with st.expander("See Internal Logic (Computer Vision Steps)"):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        _, mask = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        st.write("**Otsu Segmentation Mask (Used for Shape Analysis):**")
        st.image(mask, caption="Lesion Segmentation", width=300)
