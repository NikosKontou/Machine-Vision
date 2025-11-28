# ==========================================
# FILE 1: train_model.py
# ==========================================
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from scipy.stats import skew

# Configuration
IMAGE_FOLDER = 'data/images'
CSV_PATH = 'data/GroundTruth.csv'
MODEL_DIR = 'models'
IMG_SIZE = 224

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(limit=None):
    print("Loading Metadata...")
    df = pd.read_csv(CSV_PATH)

    classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    available_classes = [c for c in classes if c in df.columns]
    if len(available_classes) != len(classes):
        classes = available_classes

    df['target'] = df[classes].idxmax(axis=1)
    df['label_idx'] = df['target'].apply(lambda x: classes.index(x))
    df['path'] = df['image'].apply(lambda x: os.path.join(IMAGE_FOLDER, x + '.jpg'))

    if limit:
        print(f"Subsampling dataset to {limit} images for speed...")
        actual_limit = min(limit, len(df))
        df, _ = train_test_split(df, train_size=actual_limit, stratify=df['label_idx'], random_state=42)

    return df, classes


def advanced_feature_extraction(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # 1. Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    features = []

    # 2. Color Statistics
    for i in range(3):
        channel = img[:, :, i]
        features.append(np.mean(channel))
        features.append(np.std(channel))
        features.append(skew(channel.flatten()))

    # 3. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # 4. Adaptive Thresholding
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 99, 15)

    # 5. Morphological Operations
    # Step A: Opening (Erosion followed by Dilation) to remove small noise specks
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Step B: Dilation (NEW STEP) to connect nearby components (fill gaps)
    # We use a slightly larger kernel to bridge gaps
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_connected = cv2.dilate(mask_clean, kernel_dilate, iterations=2)

    # 6. Connected Components (Keep Largest)
    contours, _ = cv2.findContours(mask_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area, perimeter, compactness = 0, 0, 0

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area > 50:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)

    features.extend([area, perimeter, compactness])

    # 7. Texture
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    features.append(np.mean(gradient_magnitude))

    return np.array(features)


def basic_color_extraction(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])])


def run_experiment(df, classes):
    print("\n--- Starting Feature Extraction & Comparison ---")

    X_advanced = []
    X_basic = []
    y = []

    total = len(df)
    for idx, row in df.iterrows():
        if idx % 100 == 0: print(f"Processing {idx}/{total}...")

        feat_adv = advanced_feature_extraction(row['path'])
        feat_basic = basic_color_extraction(row['path'])

        if feat_adv is not None and feat_basic is not None:
            X_advanced.append(feat_adv)
            X_basic.append(feat_basic)
            y.append(row['label_idx'])

    X_advanced = np.array(X_advanced)
    X_basic = np.array(X_basic)
    y = np.array(y)

    if len(y) == 0:
        print("Error: No images were successfully processed.")
        return

    techniques = {
        "Basic Color (RF)": (X_basic,
                             RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        "Adv. Handcrafted (RF)": (X_advanced,
                                  RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        "Adv. Handcrafted (SVM)": (X_advanced, SVC(probability=True, class_weight='balanced', random_state=42))
    }

    best_score = 0
    best_model = None
    best_scaler = None
    best_name = ""

    for name, (features, model) in techniques.items():
        print(f"\nTraining {name}...")
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = model
            best_scaler = scaler
            best_name = name

    print(f"\nBest Model: {best_name} with Accuracy {best_score:.4f}")
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'skin_cancer_model.pkl'))
    joblib.dump(best_scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(classes, os.path.join(MODEL_DIR, 'classes.pkl'))
    print("Model saved successfully.")


if __name__ == "__main__":
    df, classes = load_data(limit=2000)
    run_experiment(df, classes)