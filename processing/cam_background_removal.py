import cv2
import numpy as np
from typing import Optional, Tuple, Any

def remove_background(
        image: np.ndarray,
        background: np.ndarray | None = None,
        min_threshold: int = 25,
        max_threshold: int = 255,
        min_area: int = 10000,
        kernel_diff: int = 35,
        krnel_fill: int = 199,
        verbose: bool = False
) -> None | list[dict[str, Any]]:
    if image is None or background is None:
        return None

    if image.shape[:2] != background.shape[:2]:
        exit("Error: Image and background must have the same dimensions.")

    gray_img = _to_grayscale(image)
    gray_bg = _to_grayscale(background)

    gray_img = cv2.GaussianBlur(gray_img, (kernel_diff,kernel_diff), 0)
    gray_bg = cv2.GaussianBlur(gray_bg, (kernel_diff,kernel_diff), 0)

    diff = cv2.absdiff(gray_img, gray_bg)

    otsu_thresh, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    final_thresh = max(min_threshold, min(int(otsu_thresh), max_threshold))

    _, mask = cv2.threshold(diff, final_thresh, 255, cv2.THRESH_BINARY)

    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (krnel_fill, krnel_fill))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    contour_data = []
    for cnt in filtered_contours:
        com_x, cov_x = _calculate_contour_metrics(cnt)
        contour_data.append({
            'contour': cnt,
            'center_of_mass_x': com_x,
            'covariance_x': cov_x
        })

    height, width = image.shape[:2]

    if verbose:
        final_mask = np.zeros_like(mask)
        cv2.drawContours(final_mask, filtered_contours, -1, 255, -1)
        masked_image = cv2.bitwise_and(image, image, mask=final_mask)
        debug_box_image = image.copy()
        for data in contour_data:
            cnt = data['contour']
            cx = int(data['center_of_mass_x'])
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(debug_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(debug_box_image, (cx, 0), (cx, height), (0, 0, 255), 2)
        cv2.imshow("Debug: Bounding Boxes", debug_box_image)
        cv2.imshow("Debug: Masked Image", masked_image)

    return contour_data

def _calculate_contour_metrics(contour: np.ndarray) -> Tuple[float, float]:
    points = contour.reshape(-1, 2).astype(float)
    center_of_mass_x = np.mean(points[:, 0])
    covariance_x = np.var(points[:, 0])
    return float(center_of_mass_x), float(covariance_x)

def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    if arr is None or arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 1: return arr[:, :, 0]
        if channels == 3: return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if channels == 4: return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported shape: {arr.shape}")

def load_background_image(background_path: str) -> Optional[np.ndarray]:
    try:
        background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
        return background
    except Exception as e:
        print(f"Error: {e}")
        return None