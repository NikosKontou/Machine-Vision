# By Kolia Aimilia, Kontoudakis Nikos, Skiada Kiki, Lampropoulou Nancy
The propejct aims to classify lessions.
## Project Structure

The project is organized as follows:

```text
ham10000_project/
│
├── data/
│   ├── images/              # Put original images here (e.g., ISIC_0024306.jpg)
│   └── GroundTruth.csv      # From Kaggle
│
├── models/                  # Created automatically
│   ├── skin_cancer_model.pkl
│   └── scaler.pkl
│
├── train_model.py           # Script 1: Training & Visualization
└── app.py                   # Script 2: Streamlit Web GUI
