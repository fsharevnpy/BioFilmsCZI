import aicspylibczi
import pathlib
from PIL import Image
import numpy as np

def process_czi(czi_path):
    print("Processing CZI file:", czi_path)
    pth = pathlib.Path(czi_path)
    
    # Check if file exists
    if not pth.exists():
        print("Error: The file does not exist.")
        return

    czi = aicspylibczi.CziFile(pth)
    print("CZI Dimensions:", czi.dims)

    img = czi.read_image(S=0, T=0, Z=0, C=0, X=1024, Y=1024)
    print("Image shape:", img.shape)

    # Example: save image
    output_image_path = "output_image.png"
    Image.fromarray(img[0, 0, 0, 0, :, :].astype(np.uint8)).save(output_image_path)
    print(f"Saved processed image to {output_image_path}")

if __name__ == "__main__":
    import sys
    process_czi(sys.argv[1])  # Get the file path from command line argument
