import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from skimage.measure import label
from skimage.morphology import skeletonize
import math

def normalize_8bit(image: np.ndarray, lower: float = 1, upper: float = 99) -> np.uint8:
    """Percentile-based 8-bit normalization with epsilon guard."""
    p_low, p_high = np.percentile(image, (lower, upper))
    scale = (p_high - p_low) + 1e-6
    img_scaled = np.clip((image - p_low) / scale * 255, 0, 255)
    return img_scaled.astype(np.uint8)


def to_uint8_slice(image_array: np.ndarray, ch: int, z: int) -> np.uint8:
    """Extract [ch, z] slice and scale to uint8 using global max as in original code."""
    origin = image_array[0, 0, 0, ch, z, :, :]
    img_max = float(image_array.max())
    if img_max <= 0:
        return np.zeros_like(origin, dtype=np.uint8)
    image = ((origin / img_max) * (2**8 - 1)).astype(np.uint8)
    return image


def denoise_and_threshold(image: np.ndarray) -> np.uint8:
    """Gaussian blur + Otsu threshold (bright cells)."""
    cell_mask = cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=5, searchWindowSize=21)
    _, thresholded = cv2.threshold(cell_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded


def find_external_contours(binary: np.ndarray) -> List[np.ndarray]:
    """Find external contours from a binary mask."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def is_closed_contour(contour: np.ndarray, threshold: float = 10.0) -> bool:
    """Check if a contour is near-closed by distance between first/last points."""
    start_point = contour[0][0]
    end_point = contour[-1][0]
    distance = np.linalg.norm(start_point - end_point)
    return distance < threshold


def closed_contour_mask(image_shape: Tuple[int, int], contours: List[np.ndarray], close_thresh: float = 10.0) -> np.uint8:
    """Fill only the closed contours to create a mask."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    for c in contours:
        if is_closed_contour(c, close_thresh):
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    return mask


def keep_max_pixel_per_contour(image: np.ndarray, contours: List[np.ndarray], close_thresh: float = 10.0) -> np.uint8:
    """For each closed contour: keep only pixels equal to the max intensity inside that contour."""
    out = np.zeros_like(image)
    for c in contours:
        if is_closed_contour(c, close_thresh):
            c_mask = np.zeros_like(image)
            cv2.drawContours(c_mask, [c], -1, 255, thickness=cv2.FILLED)
            inside = image[c_mask > 0]
            if inside.size > 0:
                m = np.max(inside)
                out[(c_mask > 0) & (image == m)] = m
    return out


def skeletonize_inside_closed_contours(image: np.ndarray, contours: List[np.ndarray], close_thresh: float = 10.0) -> Tuple[np.uint8, int]:
    """Skeletonize the region inside closed contours and count connected skeletons."""
    c_mask = closed_contour_mask(image.shape, contours, close_thresh)
    inside = cv2.bitwise_and(image, image, mask=c_mask)
    skeleton = skeletonize(inside > 0)
    skeleton_u8 = (skeleton.astype(np.uint8)) * 255
    _, num = label(skeleton_u8, connectivity=2, return_num=True)
    return skeleton_u8, num


def detect_branch_points(skeleton_u8: np.uint8) -> np.uint8:
    """Detect branch points using a 3x3 convolution (no Python loops).
    A branch point is a skeleton pixel whose 8-neighbor count > 2.
    """
    # Ensure binary 0/1
    sk = (skeleton_u8 > 0).astype(np.uint8)

    # 3x3 ones kernel counts the center + 8 neighbors
    kernel = np.ones((3, 3), dtype=np.uint8)

    # Convolution with zero padding on borders
    # neighbor_sum = count(center + neighbors); subtract center to get neighbors only
    neighbor_sum = cv2.filter2D(sk, ddepth=cv2.CV_8U, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    neighbor_cnt = neighbor_sum - sk

    # Branch point if pixel is skeleton and has more than 2 neighbors
    bp = ((sk == 1) & (neighbor_cnt > 2)).astype(np.uint8) * 255
    return bp


def overlay_skeleton(origin_gray: np.uint8, skeleton_u8: np.uint8, branch_points_u8: np.uint8 = None) -> np.ndarray:
    """Overlay skeleton (and optional branch points) on original gray image as BGR."""
    rgb = cv2.cvtColor(origin_gray, cv2.COLOR_GRAY2BGR)
    # Orange for skeleton
    rgb[skeleton_u8 > 0] = [0, 100, 255]  # B, G, R
    # Blue for branch points (optional)
    if branch_points_u8 is not None:
        rgb[branch_points_u8 > 0] = [255, 0, 0]
    return rgb


def process_channel(image_array: np.ndarray, ch: int, z: int, close_thresh: float = 10.0) -> Dict[str, Any]:
    """
    Full pipeline for one channel/z:
    - extract slice -> threshold -> contours -> keep max per contour -> skeletonize -> count -> overlays
    Returns a dict with useful outputs.
    """
    origin_u8 = to_uint8_slice(image_array, ch, z)

    th = denoise_and_threshold(origin_u8)
    contours = find_external_contours(th)

    # Keep only the brightest pixels per closed contour
    brightest = keep_max_pixel_per_contour(origin_u8, contours, close_thresh)

    # Skeletonize inside closed contours and count them
    skeleton_u8, num_skeletons = skeletonize_inside_closed_contours(brightest, contours, close_thresh)

    # Optional branch points (kept for parity with original code)
    branch_u8 = detect_branch_points(skeleton_u8)

    overlay = overlay_skeleton(origin_u8, skeleton_u8, branch_u8)

    return {
        "origin_u8": origin_u8,           # original 8-bit slice (for visualization)
        "threshold_u8": th,               # threshold mask
        "brightest_u8": brightest,        # max-per-contour mask (used later for RGB)
        "skeleton_u8": skeleton_u8,       # skeleton mask
        "branch_u8": branch_u8,           # branch points
        "overlay_bgr": overlay,           # visualization
        "num_skeletons": int(num_skeletons),
    }


def compose_rgb(dead_brightest: np.ndarray, live_brightest: np.ndarray,
                dead_count: int, live_count: int) -> np.ndarray:
    """
    Scale channels by counts as in original code and compose RGB (G for live, R for dead).
    """
    # Original logic with guard and clipping
    eps = 1e-9
    ratio = (dead_count + eps) / (live_count + eps)
    clip_ratio = np.clip(ratio, 1 / math.pi, math.pi)

    dead_scale = math.sqrt(clip_ratio)
    live_scale = 1.0 / math.sqrt(clip_ratio)

    red_raw = dead_brightest.astype(np.float32) * dead_scale
    green_raw = live_brightest.astype(np.float32) * live_scale

    rgb = np.zeros((*dead_brightest.shape, 3), dtype=np.uint8)
    rgb[..., 1] = normalize_8bit(green_raw)  # G: live
    rgb[..., 2] = normalize_8bit(red_raw)    # R: dead
    return rgb

def scale_contour(contour: np.ndarray, scale_factor: float = 1.2) -> np.ndarray:
    """Scale a contour around its centroid to include a margin around the object."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return contour  # avoid division by zero

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    pts = contour.reshape(-1, 2).astype(np.float32)
    pts[:, 0] = cx + scale_factor * (pts[:, 0] - cx)
    pts[:, 1] = cy + scale_factor * (pts[:, 1] - cy)
    return pts.round().astype(np.int32).reshape(-1, 1, 2)


def intensity_rgb(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red and green ranges in HSV
    # Red has two ranges in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Green range
    lower_green = np.array([40, 70, 50])
    upper_green = np.array([80, 255, 255])

    # Create masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Count pixels
    red_pixels = np.sum(mask_red > 0)
    green_pixels = np.sum(mask_green > 0)
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate percentages
    red_percentage = (red_pixels / total_pixels) * 100
    green_percentage = (green_pixels / total_pixels) * 100
    combined_percentage = red_percentage + green_percentage
    return combined_percentage, red_percentage, green_percentage
