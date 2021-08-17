import cv2
import os

import numpy as np

def extract_all_lines(gray_image, output_path, isdebug=False):
    img = gray_image
    (thresh, img_bin) = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)  # Thresholding the image
    if isdebug: cv2.imwrite(os.path.join(output_path, "extract_all_lines_threshold.png"), img_bin)
    img_bin_orig = img_bin.copy()
    img_bin = 255-img_bin  # Invert the image
    if isdebug: cv2.imwrite(os.path.join(output_path, "extract_all_lines_threshold_bin.png"), img_bin)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_bin = cv2.dilate(img_bin, dilate_kernel)
    if isdebug: cv2.imwrite(os.path.join(output_path, "extract_all_lines_dilate.png"), img_bin)

    scale = 0.2
    image_height, image_width = img.shape[:2]

    # Defining a kernel length
    kernel_length = min(100, max(30, int(image_width * scale)))
    verticalsize = min(100, max(30, int(image_height * scale)))
        
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    if isdebug: cv2.imwrite(os.path.join(output_path, "verticle.png"), verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    if isdebug: cv2.imwrite(os.path.join(output_path, "horizontal.png"), horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    img_final_bin = 255-img_final_bin  # Invert the image
    # no_border = np.bitwise_or(img_bin_orig, horizontal_lines_img + verticle_lines_img)
    no_border = np.bitwise_or(img_bin_orig, img_final_bin)
    (thresh, no_border) = cv2.threshold(no_border, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return no_border