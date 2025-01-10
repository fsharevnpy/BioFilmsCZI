import cv2
import numpy as np
from aicspylibczi import CziFile
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def extract_cell(image):
    background_mask = cv2.adaptiveThreshold(
        src=image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=51,
        C=2
    )

    # Invert the mask to focus on the background
    background_mask = cv2.bitwise_not(background_mask)

    # Extract the background
    background = cv2.bitwise_and(image, image, mask=background_mask)
    cell = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(background_mask))
    return background, cell

if __name__ == "__main__":
    # Read the .czi file
    czi = CziFile('/home/nguyen/BIO/BioFilmsCZI/Biofilms/image/Image_21.czi')
    image_array, _ = czi.read_image()  # image_array has shape (0, 0, 0, ch, z, h, w)

    # Select the specific channel and Z slice
    selected_channel = 0
    selected_z = 0
    origin_image = image_array[0, 0, 0, selected_channel, selected_z, :, :]
    image = ((origin_image / image_array.max()) * (pow(2,8)-1)).astype(np.uint8)

    ###############################################

    background, cell = extract_cell(image)

    # Display the results #########################

    # Count the number of zero pixels
    print("intensity: ", np.sum(cell[cell > 0]) - np.sum(background))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(background, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(cell, cmap='gray')
    plt.show()

    ###############################################
