import cv2
import numpy as np
from scipy.stats import skew
from . import config


def preprocess_image(img):
    """Standardizes image size and removes high-frequency noise."""
    if img is None: return None, None, None

    img_resized = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, config.BLUR_KERNEL, 0)

    return img_resized, img_gray, img_blur


def extract_color_stats(img):
    """Calculates Mean, Std Dev, and Skewness for R, G, B channels."""
    stats = []
    for i in range(3):
        channel = img[:, :, i]
        stats.append(np.mean(channel))
        stats.append(np.std(channel))
        stats.append(skew(channel.flatten()))
    return stats


def segment_lesion(img_blur):
    """
    Pipeline: Adaptive Threshold -> Open (Clean) -> Dilate (Connect).
    Returns intermediate masks for visualization if needed.
    """
    # 1. Adaptive Thresholding
    mask_raw = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.ADAPTIVE_BLOCK_SIZE,
        config.ADAPTIVE_C
    )

    # 2. Morphological Opening (Remove Noise)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_OPEN_KERNEL)
    mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # 3. Dilation (Connect Components)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_DILATE_KERNEL)
    mask_connected = cv2.dilate(mask_clean, kernel_dilate, iterations=2)

    return mask_raw, mask_clean, mask_connected


def isolate_largest_component(mask):
    """Filters all blobs except the largest one (the lesion)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_mask = np.zeros_like(mask)
    area, perimeter, compactness = 0, 0, 0

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        # Only keep if significant
        if area > 50:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)

    return final_mask, area, perimeter, compactness


def compute_texture_sobel(img_gray):
    """Calculates texture score using Sobel gradients."""
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    texture_score = np.mean(magnitude)

    # Return normalized visual for display purposes
    magnitude_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return texture_score, magnitude_vis


def extract_all_features_pipeline(image_path_or_array):
    """
    The Master Orchestrator.
    Can handle input as a file path (str) or a numpy array (image).
    """
    # 1. Load
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array

    if img is None: return None

    # 2. Preprocess
    img_resized, img_gray, img_blur = preprocess_image(img)

    features = []

    # 3. Color
    features.extend(extract_color_stats(img_resized))

    # 4. Segmentation Pipeline
    _, _, mask_connected = segment_lesion(img_blur)

    # 5. Shape Analysis
    _, area, perimeter, compactness = isolate_largest_component(mask_connected)
    features.extend([area, perimeter, compactness])

    # 6. Texture Analysis
    texture_score, _ = compute_texture_sobel(img_gray)
    features.append(texture_score)

    return np.array(features)