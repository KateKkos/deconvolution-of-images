import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import os

RESTORED_PATH = "/home/user/Documents/IITP RAS DS/test task 6/images/restored images (wiener)/"
SRC_PATH = "/home/user/Documents/IITP RAS DS/test task 6/images/saved gray images/"
NORM = 255.0

def psnr(src_path, src_name, rest_path, rest_name):
    source = (cv2.imread(src_path + src_name)).astype(np.float32) / 255.0
    restored = (cv2.imread(rest_path + rest_name)).astype(np.float32) / 255.0
    mse = np.mean((source - restored) ** 2)
    max_pix = 1.0

    psnr_val = 10 * np.log10((max_pix ** 2) / mse)

    return psnr_val

def my_ssim(src_path, src_name, rest_path, rest_name):
    source = (cv2.imread(src_path + src_name, cv2.IMREAD_GRAYSCALE)).astype(np.float32) / 255.0
    restored = (cv2.imread(rest_path + rest_name, cv2.IMREAD_GRAYSCALE)).astype(np.float32) / 255.0

    ssim_val = ssim(source, restored, data_range=1.0)

    return ssim_val

def get_src_name(rest_name):
    base = os.path.splitext(rest_name)[0]
    parts = base.split('_')
    scr_name = str(parts[0] + ".tiff")

    return scr_name

for rest_name in os.listdir(RESTORED_PATH):
    src_name = get_src_name(rest_name)
    psnr_val = psnr(SRC_PATH, src_name, RESTORED_PATH, rest_name)
    ssim_val = my_ssim(SRC_PATH, src_name, RESTORED_PATH, rest_name)

    print("Source: " + str(src_name) + ", restored: " + str(rest_name))
    print("PSNR: " + str(psnr_val) + ", SSIM: " + str(ssim_val))