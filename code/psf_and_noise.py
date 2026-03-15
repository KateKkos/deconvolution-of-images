# Для запукска в VS Code пропиши в его терминале: unset GTK_PATH (если не находит какой-то символ)
# Сообщения от QSocketNotifier и Gtk-Message игнорируй

import cv2
import numpy as np
from olimp.simulate.psf_gauss import PSFGauss
from olimp.processing import fft_conv, fftshift
import torch
from skimage.util import random_noise
import os
from pathlib import Path

SRC_PATH = "/home/user/Documents/IITP RAS DS/test task 6/images/raw images/"
GRAY_PATH = "/home/user/Documents/IITP RAS DS/test task 6/images/saved gray images/"
PSF_PATH = "/home/user/Documents/IITP RAS DS/test task 6/images/saved psf images/"
NOISE_PATH = "/home/user/Documents/IITP RAS DS/test task 6/images/saved noise images/"
NORM = 255.0

def img2gray_norm(path, name, normalizer):
    img = cv2.imread(path + name)

    if img is None:
        print("Image not found")
        return -1
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_float = img_gray.astype(np.float32)
    img_gray_norm = img_gray_float / normalizer

    return img_gray_norm

def show(img, normalizer):
    img_display = (img * normalizer).astype(np.uint8)
    cv2.imshow("", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    return 0

def save(img, normalizer, path, name):
    img_save = (img * normalizer).astype(np.uint8)
    save_path = path + name
    if cv2.imwrite(save_path, img_save):
        return True
    else: 
        return False

def psf(img_gr_flt_n, s_x, s_y, th):
    # fft_conv ожидает на вход тензоры с указанной размерностью (unsqueeze для указания размерности)
    img_tensor = torch.from_numpy(img_gr_flt_n).unsqueeze(0)

    H, W = img_gr_flt_n.shape
    psf_generator = PSFGauss(width=W, height=H)
    psf = psf_generator( 
        center_x = W//2,          # центр по X (задаётся таким для удобства вычислений)
        center_y = H//2,          # центр по Y
        sigma_x = s_x,            # размытие по горизонтали
        sigma_y = s_y,            # размытие по вертикали
        theta = np.radians(th)    # поворот
    )

    psf_shifted = fftshift(psf)   # Для удобства вычислений
    psf_tensor = psf_shifted.unsqueeze(0)  
    blurred_tensor = fft_conv(img_tensor, psf_tensor)   # Применяем ФРТ к изображению
    blurred_np = blurred_tensor.squeeze().numpy()

    return blurred_np

def gaussian_noise(img, std_dev):
    variance = std_dev ** 2
    noisy_img = random_noise(img, mode = "gaussian", mean = 0, var = variance, clip = True)

    return noisy_img.astype(np.float32)

    





# img_gr_flt_n = img2gray_norm(SRC_PATH, "tree.tiff", NORM)
# show(img_gr_flt_n, NORM) 
# save(img_gr_flt_n, NORM, GRAY_PATH, "tree.tiff")

psf_params = [[8, 6, 60], [2, 2, 0], [3, 5, 20], [1.5, 0.5, 255], [4, 4, 0]]

# for imgname in os.listdir(SRC_PATH):
#     img_gr_flt_n = img2gray_norm(SRC_PATH, imgname, NORM)
#     for param in psf_params:
#         blurred = psf(img_gr_flt_n, param[0], param[1], param[2])
#         base = os.path.splitext(imgname)[0]
#         blurred_name = base + "_psf_" + str(param[0]) + "_" + str(param[1]) + "_" + str(param[2]) + ".tiff"
#         save(blurred, NORM, PSF_PATH, blurred_name)
#         # show(blurred, NORM)



noises = [0.01, 0.05, 0.1]

# for imgname in os.listdir(PSF_PATH):
#     img_path = os.path.join(PSF_PATH, imgname)
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     imgname_wn_ext = os.path.splitext(imgname)[0]

#     for noise in noises:
#         img_noise = gaussian_noise(img/255, noise)
#         # show(img_noise, NORM)
#         save(img_noise, NORM, NOISE_PATH, imgname_wn_ext + "_" + str(noise) + ".tiff")

