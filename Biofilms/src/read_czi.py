import os
import sys

import numpy as np
from aicspylibczi import CziFile
import matplotlib.pyplot as plt

# Load the .czi file
if len(sys.argv) > 1 and sys.argv[1]:
    imagedir = sys.argv[1]
else:
    imagedir = "../image"

czi_files = [f for f in os.listdir(imagedir) if f.endswith('.czi')]
for file in czi_files:
    filepath = imagedir + "/" + file
    filename = file.replace(".czi","")
    filedir = "../output/" + filename
    os.makedirs(filedir, exist_ok=True)
    czi = CziFile(filepath)

    image_data, _ = czi.read_image()
    batch_dim, channel_dim, acquisition_dim, channel_count, z_stack_count, height, width = image_data.shape

    filedir_detailed = filedir + "/" + "detailed"
    os.makedirs(filedir_detailed, exist_ok=True)
    for ch in range(channel_count):
        for z in range(z_stack_count):
            sub_filename = filename + "_c" + str(ch) + "_s" + str(z)
            image = image_data[0, 0, 0, ch, z, :, :]
            plt.imsave(os.path.join(filedir_detailed, sub_filename + ".png"), image, cmap = 'gray')
            plt.close()

    filedir_combined = filedir + "/" + "combined"
    os.makedirs(filedir_combined, exist_ok=True)
    for z in range(z_stack_count):
        sub_filename = filename + "_s" + str(z)
        channel_1 = image_data[0, 0, 0, 0, z, :, :]
        channel_2 = image_data[0, 0, 0, 1, z, :, :]
        channel_1_normalized = channel_1 / np.max(channel_1)
        channel_2_normalized = channel_2 / np.max(channel_2)
        rgb_image = np.zeros((channel_1.shape[0], channel_1.shape[1], 3))
        rgb_image[..., 1] = channel_1_normalized
        rgb_image[..., 0] = channel_2_normalized
        plt.imsave(os.path.join(filedir_combined, sub_filename + ".png"), rgb_image)
        plt.close()