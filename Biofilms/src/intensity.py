import cv2
import numpy as np
from aicspylibczi import CziFile
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def scale_contour(contour, scale_factor=1.2):
    # Calculate the centroid (center of mass) of the contour
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return contour  # Avoid division by zero
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Scale the contour around its centroid
    scaled_contour = []
    for point in contour:
        x, y = point[0]
        x_scaled = cx + scale_factor * (x - cx)
        y_scaled = cy + scale_factor * (y - cy)
        scaled_contour.append([[int(x_scaled), int(y_scaled)]])
    
    return np.array(scaled_contour, dtype=np.int32)

def extract_cell(image):
    cell_mask = cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=5, searchWindowSize=21)
    _, cell_mask = cv2.threshold(cell_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint16)
    cell_mask = cv2.erode(cell_mask, kernel, iterations=1)
    cell_mask = cv2.dilate(cell_mask, kernel, iterations=10)

    contours, _ = cv2.findContours(cell_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    marked_mask = np.zeros_like(image)

    for contour in contours:
        scaled_contour = scale_contour(contour, scale_factor=1.5)
        cv2.drawContours(marked_mask, [scaled_contour], -1, 255, thickness=cv2.FILLED)

    # Extract the background
    cell = cv2.bitwise_and(image, image, mask=marked_mask)
    background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(marked_mask))

    cell_area = cv2.countNonZero(marked_mask)
    total_area = image.shape[0] * image.shape[1]
    percentage = (cell_area / total_area) * 100
    return background, cell, percentage

if __name__ == "__main__":
    # Read the .czi file
    czi = CziFile('/home/nguyen/Biofilms_git/BioFilmsCZI/Biofilms/image/Image_21.czi')
    image_array, _ = czi.read_image()  # image_array has shape (0, 0, 0, ch, z, h, w)

    selected_channel = 0
    selected_z = 3
    origin_image = image_array[0, 0, 0, selected_channel, selected_z, :, :]
    image = ((origin_image / image_array.max()) * (pow(2,8)-1)).astype(np.uint8)

    ###############################################

    background, cell, percentage = extract_cell(image)
    # Display the results #########################

    # Count the number of zero pixels
    print("intensity: ", np.sum(cell)/np.count_nonzero(cell))
    print("cell percentage: ", percentage)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(background, cmap='gray', vmin = 0, vmax = 255)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cell, cmap='gray', vmin = 0, vmax = 255)
    plt.axis("off")
    plt.show()

    ###############################################
