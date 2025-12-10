import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from . import config


def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    filename = f"{model_name.lower()}_confusion_matrix.png"
    plt.savefig(os.path.join(config.MODEL_DIR, filename))
    plt.close()


def plot_multiclass_roc(model, X_test, y_test, classes, model_name):
    """Generates and saves a Multi-class ROC Curve."""
    # 1. Binarize labels (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    n_classes = y_test_bin.shape[1]

    # 2. Get probabilities
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        print(f"{model_name} does not support probability prediction. Skipping ROC.")
        return

    # 3. Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 4. Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

    filename = f"{model_name.lower()}_roc_curve.png"
    plt.savefig(os.path.join(config.MODEL_DIR, filename))
    plt.close()


def save_classification_report(y_true, y_pred, classes, model_name):
    """Saves the text classification report."""
    report = classification_report(y_true, y_pred, target_names=classes)
    filename = f"{model_name.lower()}_report.txt"
    with open(os.path.join(config.MODEL_DIR, filename), "w") as f:
        f.write(report)
    print(f"Report saved to {filename}")