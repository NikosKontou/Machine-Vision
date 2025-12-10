import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from . import config


def plot_rf_feature_importance(model, feature_names):
    """
    Plots and saves the Feature Importance for a Random Forest model.
    Returns a DataFrame of the importance values for inspection.
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not support feature importance (not a Random Forest?). Skipping.")
        return None

    importances = model.feature_importances_
    # Sort indices in descending order
    indices = np.argsort(importances)[::-1]

    # Create Labels based on sorted indices
    sorted_names = [feature_names[i] for i in indices]

    # Plot
    plt.figure(figsize=(14, 8))
    plt.title("Random Forest: Feature Importance (Explainable AI)")
    plt.bar(range(len(importances)), importances[indices], align="center", color='teal')
    plt.xticks(range(len(importances)), sorted_names, rotation=90)
    plt.xlim([-1, len(importances)])
    plt.ylabel("Relative Importance")
    plt.tight_layout()

    save_path = os.path.join(config.MODEL_DIR, 'rf_feature_importance.png')
    plt.savefig(save_path)
    plt.close()
    print(f"XAI Plot saved to {save_path}")

    # Return Data
    return pd.DataFrame({
        'Feature': sorted_names,
        'Importance': importances[indices]
    })