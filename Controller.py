import TFL_Man
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

import matplotlib.pyplot as plt
class Controller:
    # for import data
    gt_train_path = ''
    imgs_train_path=''
    gt_val_path =''
    imgs_val_path = ''
    gt_test_path = ''
    imgs_test_path = ''
    #lists conatins the file paths
    