import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_integral_image(img):
    return np.cumsum(np.cumsum(img, axis=0), axis=1)

def rect_sum(ii, r, c, h, w):
    r2, c2 = r + h - 1, c + w - 1
    total = ii[r2, c2]
    if r > 0:
        total -= ii[r - 1, c2]
    if c > 0:
        total -= ii[r2, c - 1]
    if r > 0 and c > 0:
        total += ii[r - 1, c - 1]
    return total