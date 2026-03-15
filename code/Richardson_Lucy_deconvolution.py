import cv2
import numpy as np
from olimp.simulate.psf_gauss import PSFGauss
from olimp.processing import fftshift
import torch
import os
from skimage import restoration

NOISE_PATH = "/home/user/Documents/IITP RAS DS/test task 6/images/saved noise images/"
RESTORED_PATH = "/home/user/Documents/IITP RAS DS/test task 6/images/restored images (rl)/"
NORM = 255.0

def params_from_name(img_name):
    base = os.path.splitext(img_name)[0]  # Убирает разрешение
    parts = base.split('_') 

    sx = float(parts[2])
    sy = float(parts[3])
    theta = float(parts[4])
    std_dev = float(parts[5])

    return sx, sy, theta, std_dev

def load_img_norm_gray(path, name):
    img_gray = cv2.imread(path + name, cv2.IMREAD_GRAYSCALE)
    img_norm = img_gray.astype(np.float32) / NORM

    return img_norm

def generate_psf(h, w, sx, sy, th):
    psf_generator = PSFGauss(width = w, height = h)
    psf = psf_generator( 
        center_x = w//2,
        center_y = h//2,
        sigma_x = sx,
        sigma_y = sy,
        theta = np.radians(th)
    )

    return psf

def show(img, normalizer):
    img_display = (img * normalizer).astype(np.uint8)
    cv2.imshow("", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    return 0

def rl_deconv(path, name, sx, sy, th, iter):
    img_blurred = load_img_norm_gray(path, name)
    h, w = img_blurred.shape
    psf = generate_psf(h, w, sx, sy, th)
    psf = psf.numpy()
    psf /= np.sum(psf)
    restored = restoration.richardson_lucy(img_blurred, psf, iter)
    restored = np.clip(restored, 0, 1).astype(np.float32)
    return restored

def deconvolve(img_name, iter):
    sx, sy, th, std_dev = params_from_name(img_name)
    restored_img = rl_deconv(NOISE_PATH, img_name, sx, sy, th, iter)
    out_name = os.path.splitext(img_name)[0] + "_restored.tiff"
    cv2.imwrite(RESTORED_PATH + out_name, (restored_img * NORM).astype(np.uint8))
    # show(restored_img, NORM)
    



for imgname in os.listdir(NOISE_PATH):
    deconvolve(imgname, 10)