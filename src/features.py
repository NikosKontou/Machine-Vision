# src/features.py
import cv2
import numpy as np
from scipy.stats import skew
from . import config


def preprocess_image(img):
    """
    Standardizes, Grayscale, applies CLAHE (Smart Contrast), and Blur.
    """
    if img is None: return None, None, None, None

    img_resized = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

    # 1. Grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Instead of standard equalizeHist (which is too noisy), this enhances
    # contrast while suppressing noise (pores/hair).
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP, tileGridSize=config.CLAHE_GRID)
    img_eq = clahe.apply(img_gray)

    # 3. Blur (Applied to the EQUALIZED image)
    img_blur = cv2.GaussianBlur(img_eq, config.BLUR_KERNEL, 0)

    return img_resized, img_gray, img_eq, img_blur


def extract_color_stats(img, mask=None):
    """Calculates Mean, Std, Skew for R, G, B."""
    stats = []
    for i in range(3):
        channel = img[:, :, i]
        if mask is not None:
            pixels = channel[mask > 0]
        else:
            pixels = channel.flatten()

        if len(pixels) == 0:
            stats.extend([0, 0, 0])
        else:
            stats.append(np.mean(pixels))
            stats.append(np.std(pixels))
            stats.append(skew(pixels))
    return stats


def extract_histogram_features(img, mask=None):
    """Calculates Color Histogram for lesion area."""
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], mask, [config.HIST_BINS], [0, 256])
        cv2.normalize(hist, hist)
        hist_features.extend(hist.flatten())
    return hist_features


def segment_lesion(img_blur):
    """
    Pipeline: Otsu Thresholding -> Open (Clean) -> Dilate (Connect).
    Note: We switched from Adaptive to Otsu because CLAHE provides
    enough global contrast for Otsu to work reliably.
    """
    # 1. Otsu's Thresholding (Global)
    # Automatically finds the best split between dark lesion and light skin.
    # THRESH_BINARY_INV means "Make the dark stuff white (selected)"
    _, mask_raw = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Morphological Opening (Remove Noise)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_OPEN_KERNEL)
    mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # 3. Dilation (Connect Components)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_DILATE_KERNEL)
    mask_connected = cv2.dilate(mask_clean, kernel_dilate, iterations=2)

    return mask_raw, mask_clean, mask_connected


def isolate_largest_component(mask):
    """Filters all blobs except the largest one."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)
    area, perimeter, compactness = 0, 0, 0

    if contours:
        # Sort contours by area, descending
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Take the largest one
        cnt = sorted_contours[0]
        area = cv2.contourArea(cnt)

        # Filter: Ignore if it's practically the whole image (vignette border artifact)
        # or if it's tiny noise
        img_area = mask.shape[0] * mask.shape[1]

        if 50 < area < (img_area * 0.95):
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
        elif len(sorted_contours) > 1:
            # Fallback: If largest was essentially the whole image border, try the 2nd largest
            cnt2 = sorted_contours[1]
            area2 = cv2.contourArea(cnt2)
            if area2 > 50:
                cv2.drawContours(final_mask, [cnt2], -1, 255, -1)
                area = area2
                perimeter = cv2.arcLength(cnt2, True)
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter ** 2)

    return final_mask, area, perimeter, compactness


def compute_texture_sobel(img_gray):
    """Calculates texture score (Sobel) on RAW gray image."""
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    texture_score = np.mean(magnitude)
    magnitude_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return texture_score, magnitude_vis


def extract_all_features_pipeline(image_path_or_array):
    """Master Orchestrator."""
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array

    if img is None: return None

    # 1. Preprocess (CLAHE -> Blur)
    img_resized, img_gray, img_eq, img_blur = preprocess_image(img)

    features = []

    # 2. Segmentation (Otsu)
    _, _, mask_connected = segment_lesion(img_blur)
    mask_final, area, perimeter, compactness = isolate_largest_component(mask_connected)

    # 3. Color Analysis
    features.extend(extract_color_stats(img_resized, mask=mask_final))
    features.extend(extract_histogram_features(img_resized, mask=mask_final))

    # 4. Shape
    features.extend([area, perimeter, compactness])

    # 5. Texture
    texture_score, _ = compute_texture_sobel(img_gray)
    features.append(texture_score)

    return np.array(features)