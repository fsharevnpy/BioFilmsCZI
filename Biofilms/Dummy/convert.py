import numpy as np
import tifffile
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def convert_rgb_to_ome_tiff(image_path, output_tiff_path):
    # Load image using OpenCV (BGR format)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to RGB (OpenCV loads in BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split into R, G, B channels
    r_channel, g_channel, b_channel = cv2.split(image)

    # Stack channels as a multi-layered grayscale image (C, H, W)
    ome_tiff_data = np.stack([r_channel, g_channel, b_channel], axis=0)

    # Plot each channel separately
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    channel_names = ["Red Channel", "Green Channel", "Blue Channel"]

    for i in range(3):
        axes[i].imshow(ome_tiff_data[i], cmap="gray")
        axes[i].set_title(channel_names[i])
        axes[i].axis("off")
    plt.show()

    # Save as OME-TIFF (Zeiss-compatible)
    tifffile.imwrite(output_tiff_path, ome_tiff_data, imagej=True)

    print(f"Saved OME-TIFF: {output_tiff_path}")
    print("Now you can open it in Zeiss Zen and save as CZI.")

# Example usage
image_path = "/home/nguyen/Biofilms_git/BioFilmsCZI/Biofilms/Dummy/1_color.jpg"  # Change this to your image file
output_tiff_path = "1_color.ome.tiff"

convert_rgb_to_ome_tiff(image_path, output_tiff_path)
