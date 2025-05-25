import cv2
import math
import numpy as np
from aicspylibczi import CziFile
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import skeletonize
import math

def extract_skeleton_inside_contours(image_array, selected_channel, selected_z):
    origin_image = image_array[0, 0, 0, selected_channel, selected_z, :, :]
    image = ((origin_image / image_array.max()) * (pow(2,8)-1)).astype(np.uint8)
    origin_image = image
    #########################################################################
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use Otsu's thresholding to detect bright cells
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours of the bright cells
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(image)

    # Function to check if a contour is nearly closed
    def is_closed_contour(contour, threshold=10):
        start_point = contour[0][0]  # First point
        end_point = contour[-1][0]   # Last point
        distance = np.linalg.norm(start_point - end_point)  # Euclidean distance
        return distance < threshold  # If distance is small, it's closed

    # Fill only closed contours on the mask
    for contour in contours:
        if is_closed_contour(contour):
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)  # Fill closed contours

    # Apply the mask to keep only the inside of contours
    result = cv2.bitwise_and(image, image, mask=mask)
    denoised = result

    final_result = np.zeros_like(image)
    # Process each closed contour
    for contour in contours:
        if is_closed_contour(contour):
            # Create a mask for the specific contour
            contour_mask = np.zeros_like(image)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

            # Extract the pixels inside the contour
            inside_pixels = image[contour_mask > 0]

            if len(inside_pixels) > 0:
                # Find the highest pixel value inside the contour
                max_pixel_value = np.max(inside_pixels)

                # Keep only the highest pixel inside the contour, set others to zero
                final_result[(contour_mask > 0) & (image == max_pixel_value)] = max_pixel_value
    
    image = final_result
    #########################################################################
    # Create a mask for contour filling
    contour_mask = np.zeros_like(image)

    # Fill only closed contours on the mask
    for contour in contours:
        if is_closed_contour(contour):
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Extract the region inside the closed contours
    inside_region = cv2.bitwise_and(image, image, mask=contour_mask)

    # Convert binary image to boolean for skeletonization
    inside_region_bool = inside_region > 0  # Convert to True/False for skimage processing

    # Apply selected skeletonization method
    skeleton = skeletonize(inside_region_bool)

    # Convert skeleton to 8-bit format
    skeleton = (skeleton * 255).astype(np.uint8)

    # Label skeletons to count them
    labeled_skeletons, num_skeletons = label(skeleton, connectivity=2, return_num=True)

    # Convert grayscale image to RGB for visualization
    rgb_image = cv2.cvtColor(origin_image, cv2.COLOR_GRAY2BGR)

    # Overlay skeleton on the original image (Blue color)
    rgb_image[skeleton > 0] = [0, 0, 255]

    # Detect branch points
    branch_points = np.zeros_like(skeleton, dtype=np.uint8)
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] == 255:
                neighbors = skeleton[y-1:y+2, x-1:x+2]
                count = np.count_nonzero(neighbors) - 1
                if count > 2:  # More than 2 neighbors means a branch point
                    branch_points[y, x] = 255

    # Mark branch points in red
    rgb_image[branch_points > 0] = [0, 0, 255]

    return denoised, origin_image, num_skeletons

def normalize_8bit(img, lower=1, upper=99):
    p_low, p_high = np.percentile(img, (lower, upper))
    img_scaled = np.clip((img - p_low) / (p_high - p_low + 1e-6) * 255, 0, 255)
    return img_scaled.astype(np.uint8)

if __name__ == "__main__":
    # Read the .czi file
    czi = CziFile('/home/nguyen/Biofilms_git/BioFilmsCZI/Biofilms/image/Bi-Bag-P-P1-3h-4.czi')
    image_array, _ = czi.read_image()  # image_array has shape (0, 0, 0, ch, z, h, w)

    # Select the specific channel and Z slice
    selected_channel_arr = [0, 1]
    selected_z = 0

    result_arr = []
    for selected_channel in selected_channel_arr:
        result_arr.append(extract_skeleton_inside_contours(image_array, selected_channel, selected_z))

    ratio = result_arr[0][2]/result_arr[1][2]
    clip_ratio = np.clip(ratio, 1/math.pi, math.pi)

    dead_scale = np.sqrt(clip_ratio)
    live_scale = 1 / np.sqrt(clip_ratio)

    red_raw = result_arr[0][0] * dead_scale
    green_raw = result_arr[1][0] * live_scale

    rgb_image = np.zeros((*red_raw.shape, 3), dtype=np.uint8)
    rgb_image[..., 1] = normalize_8bit(green_raw)
    rgb_image[..., 2] = normalize_8bit(red_raw)

    cv2.imwrite("result.png", rgb_image)
    print(f"Dead {result_arr[0][2]} - Live {result_arr[1][2]}")