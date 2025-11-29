# By Kolia Aimilia, Kontoudakis Nikos, Skiada Kiki, Lampropoulou Nancy
The project aims to classify lesions.

# Project Structure
The project follows a modular structure to separate configuration, data processing, computer vision logic, and model training.

```text
ham10000_project/
│
├── data/
│   ├── images/              # Original dermoscopy images (e.g., ISIC_0024306.jpg)
│   └── GroundTruth.csv      # Metadata and labels
│
├── models/                  # Generated automatically during training
│   ├── skin_cancer_model.pkl # Trained Random Forest/SVM model
│   ├── scaler.pkl           # StandardScaler for feature normalization
│   ├── classes.pkl          # List of class names (MEL, NV, etc.)
│   └── comparison_results.png # Confusion matrix plot
│
├── src/                     # Core Logic Package
│   ├── __init__.py          # Makes this folder a Python package
│   ├── config.py            # Configuration, Constants, and Hyperparameters
│   ├── data.py              # Data loading and stratified splitting logic
│   ├── features.py          # Computer Vision pipeline (CLAHE, Otsu, Sobel, etc.)
│   └── model.py             # Model training, evaluation, and saving
│
├── train_main.py            # Script 1: Main entry point to train the model
├── app.py                   # Script 2: Streamlit Web Interface for inference
└── requirements.txt         # Project dependencies
