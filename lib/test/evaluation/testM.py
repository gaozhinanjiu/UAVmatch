import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim


def cross_correlation(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    NCC = cv2.matchTemplate(img1_gray, img2_gray, cv2.TM_CCORR_NORMED)         #归一化互相关

    MSE = np.mean((img1_gray - img2_gray) ** 2)

    SSIM = compare_ssim(img1_gray, img2_gray, multichannel=True)
    return NCC,MSE,SSIM




