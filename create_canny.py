import glob
import cv2
import numpy as np
import os
from os import path as osp


def auto_canny(image, sigma=0.05):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    lower = 100
    upper = 255
    return cv2.Canny(image, lower, upper)


def main():
    file_dir = "./images"
    canny_dir = osp.abspath(osp.join(file_dir, "../", "canny"))
    if not osp.exists(canny_dir):
        os.makedirs(canny_dir)
    image_files = glob.glob("{}/*.png".format(file_dir))
    for image_file in image_files:
        file_name = osp.basename(image_file)
        image = cv2.imread(image_file, 0)
        canny_file = osp.join(canny_dir, "canny_{}".format(file_name))
        print("Save to", canny_file)
        canny = auto_canny(image)
        # invert color of image
        canny = cv2.bitwise_not(canny)
        # Create a transparent black image with the same size as the Canny image
        alpha = np.zeros_like(canny)
        # Set the alpha channel to 0.3
        alpha[:] = 0.3 * 255
        # Merge the Canny image and the alpha channel using cv2.addWeighted()
        canny_with_alpha = cv2.merge((canny, canny, canny, alpha))
        cv2.imwrite(canny_file, canny_with_alpha)


if __name__ == "__main__":
    main()
