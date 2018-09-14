import cv2
import numpy as np
from cv2tools import vis,filt

def load_image(path, is_filename=True):
    if is_filename:
        # read image from file
        orig = cv2.imread(path)
        assert orig is not None
    else:
        # use in memory image
        orig = path
    return orig

def resize(image, maxwidth):
    cmw = max(image.shape[0:2])
    return vis.resize_perfect(image, maxwidth, maxwidth, cubic=True, a=3)

def artistic_enhance(image):
    # try penciling

    blurred = cv2.GaussianBlur(image,ksize=(0,0),sigmaX=20)

    alpha = 2.0
    beta = 0.7

    hf = (image - blurred)*alpha+beta
    return np.clip(hf, 0, 1)
