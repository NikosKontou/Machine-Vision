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
ADAPTIVE_BLOCK_SIZE = 99
ADAPTIVE_C = 15
BLUR_KERNEL = (9, 9)
MORPH_OPEN_KERNEL = (3, 3)
MORPH_DILATE_KERNEL = (5, 5)

# Legend for UI
LEGEND_DATA = {
    "Abbreviation": ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
    "Full Diagnosis": [
        "Melanoma", "Melanocytic nevi", "Basal cell carcinoma",
        "Actinic keratoses", "Benign keratosis-like lesions",
        "Dermatofibroma", "Vascular lesions"
    ],
    "Description": [
        "Malignant skin tumor (Cancerous).", "Benign moles.",
        "Common skin cancer.", "Pre-cancerous lesions.",
        "Non-cancerous growths.", "Benign nodules.", "Benign blood vessel lesions."
    ]
}