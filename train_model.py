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
IMG_SIZE = 224  # Resize per Lecture 2

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(limit=None):
    """
    Loads data and converts One-Hot labels to single column.
    """
    print("Loading Metadata...")
    df = pd.read_csv(CSV_PATH)

    # Map One-Hot encoded columns to a single target class
    # FIXED: Added the specific column names for HAM10000
    classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    # Check if these columns actually exist in the CSV, otherwise adjust
    available_classes = [c for c in classes if c in df.columns]
    if len(available_classes) != len(classes):
        print(f"Warning: Expected {classes} but found {available_classes}")
        classes = available_classes

    df['target'] = df[classes].idxmax(axis=1)
    df['label_idx'] = df['target'].apply(lambda x: classes.index(x))

    # Add extension if missing
    df['path'] = df['image'].apply(lambda x: os.path.join(IMAGE_FOLDER, x + '.jpg'))

    # Stratified sampling for faster testing if limit is set
    if limit:
        print(f"Subsampling dataset to {limit} images for speed...")
        # Handle cases where limit > dataset size
        actual_limit = min(limit, len(df))
        df, _ = train_test_split(df, train_size=actual_limit, stratify=df['label_idx'], random_state=42)

    return df, classes


def advanced_feature_extraction(image_path):
    """
    Implements methodologies: Resize, Otsu, Sobel, Color Histograms
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # 1. Resize (Lecture 2)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    features = []

    # 2. Color Statistics (Lecture 4 - Histograms)
    # Extract Mean, Std, Skewness for R, G, B
    for i in range(3):
        channel = img[:, :, i]
        features.append(np.mean(channel))
        features.append(np.std(channel))
        features.append(skew(channel.flatten()))

    # 3. Preprocessing for Segmentation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur to remove noise (Lecture 4)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Otsu's Thresholding (Lecture 8)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 5. Shape Features via Connected Components (Lecture 6/7)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find largest contour (assumed to be the lesion)
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        # Compactness
        if area > 0 and perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter ** 2)
        else:
            compactness = 0
    else:
        area, perimeter, compactness = 0, 0, 0

    features.extend([area, perimeter, compactness])

    # 6. Texture via Sobel Edge Detection (Lecture 8)
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

    # FIXED: Initialize lists
    X_advanced = []
    X_basic = []
    y = []

    # Process images
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
        print("Error: No images were successfully processed. Check image paths.")
        return

    # Compare Classifiers
    techniques = {
        "Basic Color (RF)": (X_basic,
                             RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        "Adv. Handcrafted (RF)": (X_advanced,
                                  RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        "Adv. Handcrafted (SVM)": (X_advanced, SVC(probability=True, class_weight='balanced', random_state=42))
    }

    results = {}
    best_model = None
    best_scaler = None
    best_score = 0
    best_name = ""

    # Visualization Setup
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (name, (features, model)) in enumerate(techniques.items()):
        print(f"\nTraining {name}...")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, stratify=y, random_state=42)

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"Accuracy: {acc:.4f}")

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], xticklabels=classes, yticklabels=classes)
        axes[i].set_title(f"{name}\nAcc: {acc:.2f}")
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

        # Save Best
        if acc > best_score:
            best_score = acc
            best_model = model
            best_scaler = scaler
            best_name = name

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'comparison_results.png'))
    print("\nComparison plot saved to models/comparison_results.png")

    # Save best model artifacts
    print(f"\nBest Model: {best_name} with Accuracy {best_score:.4f}")
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'skin_cancer_model.pkl'))
    joblib.dump(best_scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(classes, os.path.join(MODEL_DIR, 'classes.pkl'))
    print("Model saved successfully.")


if __name__ == "__main__":
    # LIMIT=2000 for faster testing. Set LIMIT=None for full dataset.
    df, classes = load_data(limit=2000)
    run_experiment(df, classes)