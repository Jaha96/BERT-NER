
import concurrent.futures
import logging
import multiprocessing as mp
import os
import platform
import shutil
import time

import cv2
import numpy as np
import pandas as pd
from google.cloud import vision
from numpy.lib.function_base import average
from scipy.stats import norm

from line_process import extract_all_lines
from vision_functions import (fix_orientation, group_rows, img2texts,
                              text_annotation2format)


def clear_folder(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as exception:
            logging.error('Failed to delete %s. Reason: %s' % (file_path, exception))

def create_new_dir(dir):
    try:
        if os.path.exists(dir):
            clear_folder(dir)
        else:
            os.makedirs(dir)
    except Exception as e:
        print(e)

def get_file_name(file_name: str):
    # Extracting file name from the path
    try:
        file_name = os.path.basename(file_name).split('.')[0]    
        # file_name = file_name.split()[0]    
    except Exception:
        return file_name
    return file_name

def get_text(borderless_image, output_path, fname, vision_client, is_debug = False):
    top_vision_response = img2texts(borderless_image, vision_client)
    top_text_anno, top_height_average = text_annotation2format(top_vision_response, borderless_image, fname + "_9_vision_formatted.png", output_path, is_debug)
    top_rows_grouped_text = group_rows(top_text_anno, top_height_average, borderless_image, output_path, fname + "_9_vision_formatted.png", is_debug)

    all_text = ""
    for row in top_rows_grouped_text:
        for col in row:
            all_text += col.text

    return all_text



def remove_borders(gray_image, output_path, fname, is_debug = False):

    orig_image = gray_image.copy()

    # 1. Clear borders using morphological operations
    no_borders = extract_all_lines(gray_image, output_path, is_debug)
    if is_debug: cv2.imwrite(os.path.join(output_path, fname+"_1_borderless.png"), no_borders)
    return no_borders


def start_processing(image_path):
    is_debug = False
    start = time.time()
    # 1496c861-4ed7-4c37-9eef-cdc10b443e2f.json
    # summit202109-904a442b8d41.json
    vision_cred_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '1496c861-4ed7-4c37-9eef-cdc10b443e2f.json'))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = vision_cred_path

    vision_client = vision.ImageAnnotatorClient()
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(APP_ROOT, 'output_path')
    # image_path = "C:/work/image2text/test_images/tegaki.png"
    fname = get_file_name(image_path)

    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    curr_file_output_path = os.path.join(output_path, fname)
    create_new_dir(curr_file_output_path)

    vision_response = img2texts(gray_image, vision_client)
    orientation_fixed_image = fix_orientation(gray_image, vision_response)
    orientation_fixed_src = os.path.join(curr_file_output_path, fname + "_orientation_fixed.png")
    cv2.imwrite(orientation_fixed_src, orientation_fixed_image)

    borderless_image = remove_borders(orientation_fixed_image, curr_file_output_path, fname, is_debug)
    text_of_image = get_text(borderless_image, curr_file_output_path, fname, vision_client, is_debug)
    print(text_of_image)

    print("Total elapsed time:")
    end = time.time()
    print(end - start)

    return text_of_image