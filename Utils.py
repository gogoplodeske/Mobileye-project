import os
import glob
import cv2
import os
import random
import numpy as np
from scipy import misc
import sys
from random import shuffle
from random import uniform
import zipfile
from collections import OrderedDict
import glob
import time
from PIL import Image, ImageFilter
from scipy.signal import convolve2d
from scipy.ndimage.filters import maximum_filter
import scipy.ndimage as ndimage
import scipy
import matplotlib.pyplot as plt


def get_files(imgs_dir, gt_dir):
    cities = os.listdir(imgs_dir)
    gt = []
    imgs = []
    for city in cities:
        new_gt_path = os.path.join(gt_dir, city)
        new_imgs_path = os.path.join(imgs_dir, city)
        gt += glob.glob(os.path.join(new_gt_path, "*labelIds.png"))
        imgs += glob.glob(os.path.join(new_imgs_path, "*.png"))
    imgs.sort()
    gt.sort()
    return imgs, gt

def Read_image(img_pth ,im_size=500):
    # Crop the size we want from a random spot in the image (as a form of
    # minor data augmentation)
    shape_im = misc.imread(img_pth)
    new_start_row = np.random.randint(0, shape_im.shape[0] - im_size)
    new_start_col = np.random.randint(0, shape_im.shape[1] - im_size)
    train_im = misc.imread(img_pth).astype(np.float32)
    train_im = train_im[new_start_row:new_start_row + im_size, new_start_col:new_start_col + im_size]
    return train_im


def minimizing(mask):
    maskorg= np.copy(mask)
    for i in range(2, mask.shape[0]):
        for j in range(2, mask.shape[1]):
            if (True in maskorg[i-2:i, j-2:j+1]) or (True in maskorg[i-2:i+1, j-2:j]):
                mask[i,j] = False
    return mask


def findcandidate(currentframe,img, c, org):
    kernel = np.ones((7, 7), np.float32) / (-49)
    kernel[1, 1] = 48 / 49
    # dst = cv2.filter2D(img,-1,kernel)
    kernel2 = np.ones((3, 3), np.float32) / (9)
    lowpass = convolve2d(img, kernel2, 'same')
    high = convolve2d(img, kernel, 'same')

    dst = scipy.ndimage.maximum_filter(high, 20)
    # plt.subplot(121), plt.imshow(img * 255, c), plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(dst ** 2, cmap=c), plt.title('Averaging')
    # plt.xticks([]), plt.yticks([])
    # plt.colorbar();
    # plt.show()
    mask = dst ** 2 > 6000
    mask2 = mask.copy()
    minimizing(mask)
    res = 255 * mask # removing to dense canidates
    res1 = 255 * mask2 # not minimized full candidate
    if c == 'Reds':
        mask3 = np.stack((mask2, np.zeros(np.shape(mask), dtype=bool), np.zeros(np.shape(mask), dtype=bool)), axis=2)
        currentframe['AuxRed'] = res1
        currentframe['xRed'] = np.nonzero(res)[0]
        currentframe['yRed'] = np.nonzero(res)[1]

    else:
        mask3 = np.stack((np.zeros(np.shape(mask), dtype=bool), mask2, np.zeros(np.shape(mask), dtype=bool)), axis=2)
        currentframe['AuxGreen'] = res1
        currentframe['xGreen'] = np.nonzero(res)[0]
        currentframe['yGreen'] = np.nonzero(res)[1]
    org[mask3] = 255
    # plt.imshow(res)
    # plt.colorbar();
    # plt.show()
    # plt.imshow(res1)
    # plt.colorbar();
    # plt.show()
    return org
