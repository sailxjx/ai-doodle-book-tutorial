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
    lower = 200
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
        cv2.imwrite(canny_file, canny)

if __name__ == "__main__":
    main()
