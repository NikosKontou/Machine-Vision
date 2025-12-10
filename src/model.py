import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from . import config, plots, explainability, features


def train_and_evaluate(X, y, classes):
    """
    Trains models, generates extensive performance plots (ROC, CM),
    creates Explainable AI plots (Feature Importance), and saves the best model.
    """

    # 1. Define Models
    techniques = {
        "RF": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "SVM": SVC(probability=True, class_weight='balanced', random_state=42)
    }

    best_score = 0
    best_model = None
    best_scaler = None
    best_name = ""

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 2. Split Data
    print(f"Splitting data (Total samples: {len(y)})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 3. Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. Training Loop
    for name, model in techniques.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        acc = accuracy_score(y_test, preds)

        print(f"--> {name} Accuracy: {acc:.4f}")

        # --- NEW: PLOTTING METRICS ---
        print(f"Generating ROC and Confusion Matrix for {name}...")
        plots.plot_confusion_matrix(y_test, preds, classes, name)
        plots.plot_multiclass_roc(model, X_test_s, y_test, classes, name)
        plots.save_classification_report(y_test, preds, classes, name)

        # --- NEW: EXPLAINABLE AI (RF Only) ---
        if name == "RF":
            print("Generating Feature Importance Plot (XAI)...")
            feature_names = features.get_feature_names()

            # Check for length mismatch (Just in case config changed)
            if len(feature_names) != X_train.shape[1]:
                print(f"Warning: Name count ({len(feature_names)}) != Feature count ({X_train.shape[1]})")
                # Fallback to generic names if mismatch
                feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

            explainability.plot_rf_feature_importance(model, feature_names)

        # Track Best
        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name
            best_scaler = scaler

            # Save Artifacts
    print(f"\nSaving Best Model: {best_name}")
    joblib.dump(best_model, os.path.join(config.MODEL_DIR, 'skin_cancer_model.pkl'))
    joblib.dump(best_scaler, os.path.join(config.MODEL_DIR, 'scaler.pkl'))
    joblib.dump(classes, os.path.join(config.MODEL_DIR, 'classes.pkl'))