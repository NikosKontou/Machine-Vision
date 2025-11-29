import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from . import config


def train_and_evaluate(X, y, classes):
    """Trains models, compares them, generates plots, and saves the best one."""

    techniques = {
        "RF": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "SVM": SVC(probability=True, class_weight='balanced', random_state=42)
    }

    best_score = 0
    best_model = None
    best_scaler = None
    best_name = ""

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, len(techniques), figsize=(12, 5))

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scaling is mandatory for SVM, good for RF
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    for i, (name, model) in enumerate(techniques.items()):
        print(f"Training {name}...")
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        acc = accuracy_score(y_test, preds)

        print(f"--> {name} Accuracy: {acc:.4f}")

        # Plot
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=classes, yticklabels=classes)
        axes[i].set_title(f"{name} (Acc: {acc:.2f})")

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name
            best_scaler = scaler  # Save the scaler fitted on training data

    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_DIR, 'comparison_results.png'))

    # Save Artifacts
    print(f"\nSaving Best Model: {best_name}")
    joblib.dump(best_model, os.path.join(config.MODEL_DIR, 'skin_cancer_model.pkl'))
    joblib.dump(best_scaler, os.path.join(config.MODEL_DIR, 'scaler.pkl'))
    joblib.dump(classes, os.path.join(config.MODEL_DIR, 'classes.pkl'))