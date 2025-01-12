import cv2
import numpy as np
from aicspylibczi import CziFile
import matplotlib.pyplot as plt

def detect_circles(image, dp=1.2, minD=5, param1=25, param2=6, start_radius=2, step=3):
    raw_circles = []
    radius = start_radius

    while True:
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minD,
            param1=param1,
            param2=param2,
            minRadius=radius,
            maxRadius=radius + step
        )
        
        if circles is not None:
            raw_circles.append(circles)
            if len(circles[0]) == 1:
                break
        
        minD += step
        radius += step
        if radius > image.shape[0] // 2:
            break

    # Concatenate results
    if raw_circles:
        return np.concatenate(raw_circles, axis=1)
    return None

if __name__ == "__main__":
    # Read the .czi file
    czi = CziFile('/home/nguyen/Biofilms_git/BioFilmsCZI/Biofilms/image/Image_21.czi')
    image_array, _ = czi.read_image()  # image_array has shape (0, 0, 0, ch, z, h, w)

    # Select the specific channel and Z slice
    selected_channel = 1
    selected_z = 0
    origin_image = image_array[0, 0, 0, selected_channel, selected_z, :, :]
    image = ((origin_image / image_array.max()) * (pow(2,8)-1)).astype(np.uint8)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Non-Local Means Denoising
    image = cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=5, searchWindowSize=21)

    # erosion
    kernel = np.ones((2, 2), np.uint16)
    image = cv2.erode(image, kernel, iterations=1)

    # dilation
    kernel = np.ones((2, 2), np.uint16)
    image = cv2.dilate(image, kernel, iterations=3)

    # Setup for Hough Circles
    raw_circles = detect_circles(image)

    # Item1: Check if the surface is too dark
    pass1_circles = []
    if raw_circles[0] is not None:
        circles = np.round(raw_circles[0]).astype('int')
        for (x, y, r) in circles:
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:            
                # Draw the circle and mark the center
                y_, x_ = np.ogrid[:image.shape[0], :image.shape[1]]
                mask = (x_ - x)**2 + (y_ - y)**2 <= r**2
                if np.mean(image[mask]) > (pow(2,8)/((1+np.sqrt(5))/2)):
                    pass1_circles.append([x,y,r])

    # Item2: Remove too closed circles
    pass2_circles = []
    for circle in pass1_circles:
        x, y, r = circle
        close = False

        for cx, cy, cr in pass2_circles:
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if distance < (r+cr)/2:
                close = True
                break

        if not close:
            pass2_circles.append(circle)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # FIX ME: image --> origin_image
    for c in pass2_circles:
        cv2.circle(image, (c[0], c[1]), c[2], (0, pow(2,16)-1, 0), 0)
    
    # Save and display the result
    output_path = 'watershed_result.png'
    cv2.imwrite(output_path, image)

    # Print confirmation
    print(f"Number of cells: {len(pass2_circles)}")

    # FIX ME: Test only
    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap='gray', vmin = 0, vmax = 255)
    # plt.hist(image[image>0].ravel(), bins=256, range=(0, 256), color='gray')
    plt.show()
