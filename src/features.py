import cv2
import numpy as np
from scipy.stats import skew
from . import config


def get_feature_names():
    """
    Returns the list of feature names in the exact order they are extracted
    by the pipeline. Used for Explainable AI plots.
    """
    names = []

    # 1. Color Stats (Mean, Std, Skew for B, G, R)
    # OpenCV loads images as BGR
    for c in ['Blue', 'Green', 'Red']:
        names.extend([f'{c}_Mean', f'{c}_Std', f'{c}_Skew'])

    # 2. Histogram (Bins for B, G, R)
    for c in ['Blue', 'Green', 'Red']:
        for i in range(config.HIST_BINS):
            names.append(f'{c}_Hist_Bin_{i}')

    # 3. Shape
    names.extend(['Area', 'Perimeter', 'Compactness'])

    # 4. Texture
    names.append('Texture_EdgeDensity')

    return names


def resize_with_padding(img, target_size=config.IMG_SIZE):
    """
    Resizes image while preserving aspect ratio, then pads to square.
    Best practice for HAM10000 (600x450) to avoid geometric distortion.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    pad_w = target_size - new_w
    pad_h = target_size - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    return padded


def preprocess_image(img):
    """Standardizes, Grayscale, CLAHE, and Blur."""
    if img is None: return None, None, None, None

    img_resized = resize_with_padding(img, target_size=config.IMG_SIZE)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP, tileGridSize=config.CLAHE_GRID)
    img_eq = clahe.apply(img_gray)

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
    """Pipeline: Otsu Thresholding -> Open (Clean) -> Dilate (Connect)."""
    _, mask_raw = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_OPEN_KERNEL)
    mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel_open, iterations=2)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_DILATE_KERNEL)
    mask_connected = cv2.dilate(mask_clean, kernel_dilate, iterations=2)

    return mask_raw, mask_clean, mask_connected


def isolate_largest_component(mask):
    """Filters all blobs except the largest one."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)
    area, perimeter, compactness = 0, 0, 0

    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cnt = sorted_contours[0]
        area = cv2.contourArea(cnt)

        img_area = mask.shape[0] * mask.shape[1]

        if 50 < area < (img_area * 0.95):
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
        elif len(sorted_contours) > 1:
            cnt2 = sorted_contours[1]
            area2 = cv2.contourArea(cnt2)
            if area2 > 50:
                cv2.drawContours(final_mask, [cnt2], -1, 255, -1)
                area = area2
                perimeter = cv2.arcLength(cnt2, True)
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter ** 2)

    return final_mask, area, perimeter, compactness


def compute_texture_canny(img_gray, mask=None):
    """Calculates texture score using Canny Edge Detection."""
    edges = cv2.Canny(img_gray, 100, 200)

    if mask is not None:
        lesion_edges = edges[mask > 0]
        if len(lesion_edges) > 0:
            texture_score = np.mean(lesion_edges)
        else:
            texture_score = 0
    else:
        texture_score = np.mean(edges)

    edges_vis = edges.copy()
    if mask is not None:
        edges_vis = cv2.bitwise_and(edges_vis, edges_vis, mask=mask)

    return texture_score, edges_vis


def extract_all_features_pipeline(image_path_or_array):
    """Master Orchestrator."""
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array

    if img is None: return None

    # 1. Preprocess
    img_resized, img_gray, img_eq, img_blur = preprocess_image(img)

    features = []

    # 2. Segmentation
    _, _, mask_connected = segment_lesion(img_blur)
    mask_final, area, perimeter, compactness = isolate_largest_component(mask_connected)

    # 3. Color Analysis
    features.extend(extract_color_stats(img_resized, mask=mask_final))
    features.extend(extract_histogram_features(img_resized, mask=mask_final))

    # 4. Shape
    features.extend([area, perimeter, compactness])

    # 5. Texture
    texture_score, _ = compute_texture_canny(img_gray, mask=mask_final)
    features.append(texture_score)

    return np.array(features)