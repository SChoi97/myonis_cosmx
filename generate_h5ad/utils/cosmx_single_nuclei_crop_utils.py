import re
from typing import Tuple

import cv2
import numpy as np


def safe_str(val) -> str:
    """Sanitize strings for filenames (keep A-Z a-z 0-9 _ . -)."""
    s = str(val) if val is not None else ""
    s = s.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "", s)


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 RGB (supports grayscale or RGBA)."""
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3]
    raise ValueError(f"Unsupported image shape {img.shape}; expected HxW or HxWx3/4")


def center_object(obj_img: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Translate object so its centroid is at the image center."""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # Fallback: center stays unchanged
        return obj_img, contour

    shift_x = obj_img.shape[1] // 2 - cx
    shift_y = obj_img.shape[0] // 2 - cy

    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered_img = cv2.warpAffine(obj_img, translation_matrix, (obj_img.shape[1], obj_img.shape[0]))
    centered_contour = contour + np.array([shift_x, shift_y])
    return centered_img, centered_contour


def align_object(obj_img: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Rotate object so its fitted ellipse major axis is vertical."""
    ellipse = cv2.fitEllipse(contour)
    angle = ellipse[2]
    if angle > 90:
        angle -= 180
    center = (obj_img.shape[1] // 2, obj_img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_obj = cv2.warpAffine(obj_img, rotation_matrix, obj_img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_obj


def pad_to_size(img: np.ndarray, target_height: int = 512, target_width: int = 512) -> np.ndarray:
    """Pad an image to a specific size while keeping content centered."""
    height, width = img.shape[:2]
    pad_top = max((target_height - height) // 2, 0)
    pad_bottom = max(target_height - height - pad_top, 0)
    pad_left = max((target_width - width) // 2, 0)
    pad_right = max(target_width - width - pad_left, 0)
    padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    return padded_img


def generate_aligned_crop(
    image: np.ndarray, contour_points: np.ndarray, canvas_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a centered, major-axis-aligned crop and matching binary mask from a contour.
    Mirrors the behavior used in generate_single_cell_crop for a single object.
    """
    contour = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))
    h, w = image.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    object_img = cv2.bitwise_and(image, image, mask=mask)

    centered_obj, centered_contour = center_object(object_img, contour)
    aligned_obj = align_object(centered_obj, centered_contour)
    padded_obj = pad_to_size(aligned_obj, target_height=canvas_size, target_width=canvas_size)

    centered_mask, centered_contour = center_object(mask, contour)
    aligned_mask = align_object(centered_mask, centered_contour)
    padded_mask = pad_to_size(aligned_mask, target_height=canvas_size, target_width=canvas_size)

    binary_mask = ((padded_mask > 0).astype(np.uint8)) * 255  # 0/255 for 8-bit PNG
    return padded_obj.astype(np.uint8), binary_mask

