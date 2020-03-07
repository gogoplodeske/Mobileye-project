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
import Utils
import matplotlib.pyplot as plt

class Controller:
    gt_dir = ''
    imgs_dir=''
    batch_size = 10
    train_imgs =[]
    train_gt = []
    val_imgs=[]
    val_gt = []
    test_imgs=[]
    test_gt = []
    result_dir_part1=''
    result_dir_part2 = ''
    result_dir_part3 = ''
    def __init__(self,gtdir,imgsdir,batch,dir_part1 ):
        #gt_dir = r'C:\Users\RENT\Desktop\Mobilye project\CityScapes\gtFine'
        #imgs_dir = r'C:\Users\RENT\Desktop\Mobilye project\CityScapes\leftImg8bit'
        self.gt_dir = gtdir
        self.imgs_dir = imgsdir
        self.batch_size = batch
        self.result_dir_part1 = dir_part1
        # Import data
        gt_train_path = os.path.join(self.gt_dir, 'train')
        imgs_train_path = os.path.join(self.imgs_dir, 'train')
        gt_val_path = os.path.join(self.gt_dir, 'val')
        imgs_val_path = os.path.join(self.imgs_dir, 'val')
        gt_test_path = os.path.join(self.gt_dir, 'test')
        imgs_test_path = os.path.join(self.imgs_dir, 'test')
        self.train_imgs, self.train_gt = Utils.get_files(imgs_train_path, gt_train_path)
        print(len(self.train_imgs))
        self.val_imgs, self.val_gt = Utils.get_files(imgs_val_path, gt_val_path)
        self.test_imgs, self.test_gt = Utils.get_files(imgs_test_path, gt_test_path)
    def run(self):
        if self.batch_size == 0:
            return
        tfl_man = TFL_Man.TFL_Man(0,self.train_imgs[0])
        for i in range(self.batch_size):
            tfl_man.Part1(self.result_dir_part1)
            k = i+1
            if k !=self.batch_size:
                tfl_man.Setframe(k,self.train_imgs[k])





    # Get training data filenames
    # for import data
    # gt_train_path = ''
    # imgs_train_path=''
    # gt_val_path =''
    # imgs_val_path = ''
    # gt_test_path = ''
    # imgs_test_path = ''
    # #lists conatins the file paths
    # train_imgs =[]
    # train_gt = []
    # val_imgs=[]
    # val_gt = []
    # test_imgs=[]
    # test_gt = []
    # def __init__(self, frameID,framepath):
    #     super().__init__()


