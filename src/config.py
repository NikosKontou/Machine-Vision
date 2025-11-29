# src/config.py
import os

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_FOLDER = os.path.join(base_dir, 'data', 'images')
CSV_PATH = os.path.join(base_dir, 'data', 'GroundTruth.csv')
MODEL_DIR = os.path.join(base_dir, 'models')

# Constants
IMG_SIZE = 224
CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# Hyperparameters
BLUR_KERNEL = (9, 9)
MORPH_OPEN_KERNEL = (5, 5)   # Increased slightly to remove more noise
MORPH_DILATE_KERNEL = (5, 5)

# CLAHE Settings (Smart Equalization)
CLAHE_CLIP = 2.0             # Threshold for contrast limiting
CLAHE_GRID = (8, 8)          # Grid size for local equalization

# Histogram Config
HIST_BINS = 8

# Legend (Same as before)
LEGEND_DATA = {
    "Abbreviation": ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
    "Full Diagnosis": ["Melanoma", "Melanocytic nevi", "Basal cell carcinoma", "Actinic keratoses", "Benign keratosis-like lesions", "Dermatofibroma", "Vascular lesions"],
    "Description": ["Malignant skin tumor.", "Benign moles.", "Common skin cancer.", "Pre-cancerous lesions.", "Non-cancerous growths.", "Benign nodules.", "Benign blood vessel lesions."]
}