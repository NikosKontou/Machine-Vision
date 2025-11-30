import cv2
import numpy as np
from scipy.stats import skew
from . import config


def resize_with_padding(img, target_size=224):
    """
    Resizes image while preserving aspect ratio.
    Best practice to avoid distortion.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with proper interpolation
    resized = cv2.resize(
        img, (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    )

    return resized


def preprocess_image(img):
    """
    Standardizes (with padding), Grayscale, applies CLAHE (Smart Contrast), and Blur.
    """
    if img is None: return None, None, None, None

    # 1. Resize with Padding (Preserves Aspect Ratio)
    img_resized = resize_with_padding(img, target_size=config.IMG_SIZE)

    # 2. Grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP, tileGridSize=config.CLAHE_GRID)
    img_eq = clahe.apply(img_gray)

    # 4. Blur (Applied to the EQUALIZED image)
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
    """
    # 1. Otsu's Thresholding (Global)
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

        # Vignette/Border guard
        img_area = mask.shape[0] * mask.shape[1]

        if 50 < area < (img_area * 0.95):
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
        elif len(sorted_contours) > 1:
            # Fallback to 2nd largest
            cnt2 = sorted_contours[1]
            area2 = cv2.contourArea(cnt2)
            if area2 > 50:
                cv2.drawContours(final_mask, [cnt2], -1, 255, -1)
                area = area2
                perimeter = cv2.arcLength(cnt2, True)
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter ** 2)

    return final_mask, area, perimeter, compactness


def compute_texture_sobel(img_gray, mask=None):
    """
    Calculates texture score (Sobel) on RAW gray image.
    Uses mask to restrict the MEAN calculation to the lesion area only.
    """
    # 1. Calculate Gradients on the FULL image first.
    # We do this on the full image so that the kernel (3x3) works correctly 
    # at the boundaries of the lesion. If we masked the image to black first,
    # the Sobel would detect massive artificial edges at the mask border.
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 2. Use Mask for Calculation
    if mask is not None:
        # Only select the gradient magnitude pixels that fall inside the lesion mask
        lesion_gradients = magnitude[mask > 0]

        if len(lesion_gradients) > 0:
            texture_score = np.mean(lesion_gradients)
        else:
            texture_score = 0
    else:
        texture_score = np.mean(magnitude)

    # Normalize for visualization purposes (0-255)
    magnitude_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply mask to visualization if provided
    if mask is not None:
        magnitude_vis = cv2.bitwise_and(magnitude_vis, magnitude_vis, mask=mask)

    return texture_score, magnitude_vis


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

    # 5. Texture (Passed mask_final here)
    texture_score, _ = compute_texture_sobel(img_gray, mask=mask_final)
    features.append(texture_score)

    return np.array(features)