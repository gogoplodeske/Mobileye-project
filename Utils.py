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
import imutils
from PIL import Image, ImageFilter
from scipy.signal import convolve2d
from scipy.ndimage.filters import maximum_filter
import scipy.ndimage as ndimage
import scipy
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import pickle
from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
import math


def get_files(imgs_dir, gt_dir):
    #  cities = os.listdir(imgs_dir)
    gt = []
    imgs = []
    # for city in cities:
    new_gt_path = os.path.join(gt_dir, )
    new_imgs_path = os.path.join(imgs_dir)
    gt += glob.glob(os.path.join(new_gt_path, "*labelIds.png"))
    imgs += glob.glob(os.path.join(new_imgs_path, "*.jpg"))
    imgs.sort()
    gt.sort()
    return imgs, gt


def Read_image(img_pth, im_size=500):
    # Crop the size we want from a random spot in the image (as a form of
    # minor data augmentation)
    shape_im = misc.imread(img_pth)
    new_start_row = np.random.randint(0, shape_im.shape[0] - im_size)
    new_start_col = np.random.randint(0, shape_im.shape[1] - im_size)
    train_im = misc.imread(img_pth).astype(np.float32)
    # train_im = train_im[new_start_row:new_start_row + im_size, new_start_col:new_start_col + im_size]
    return train_im


def minimizing(mask):
    maskorg = np.copy(mask)
    for i in range(40, mask.shape[0]):
        for j in range(40, mask.shape[1]):
            if (1 in maskorg[i - 40:i, j - 40:j + 1]) or (1 in maskorg[i - 40:i + 1, j - 40:j]):
                mask[i, j] = 0
    return mask


def find_maxima(img, threshold=80, roi=40):
    size = 2 * roi + 1
    image = img.copy()
    image_max = ndimage.maximum_filter(image, size=size, mode='constant')
    mask = (image == image_max)
    image *= mask
    # plt.subplot(121), plt.imshow(image_max, cmap='Reds'), plt.title('mask')
    # plt.xticks([]), plt.yticks([])
    # plt.colorbar();
    # plt.subplot(122), plt.imshow(image, cmap='Reds'), plt.title('image')
    # plt.xticks([]), plt.yticks([])
    # plt.colorbar();
    # plt.show()
    # Remove the image borders
    image[:roi] = 0
    image[-roi:] = 0
    image[:, :roi] = 0
    image[:, -roi:] = 0

    # Optionally find peaks above some threshold
    image_t = (img > threshold) * 1

    return image * image_t


def findcandidate(currentframe, img, invimg, bimg, c, org):
    kernel = np.ones((41, 41), np.float32) / (-1681)
    kernel[20, 20] = 1680 / 1681
    # dst = cv2.filter2D(img,-1,kernel)
    kernel2 = np.ones((9, 9), np.float32) / (81)
    lowpass = convolve2d(img, kernel2, 'same')
    high = convolve2d(img, kernel, 'same')

    dst = find_maxima(high)
    # plt.subplot(121), plt.imshow(img * 255, c), plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.colorbar();
    # plt.subplot(122), plt.imshow(dst , cmap=c), plt.title('Averaging')
    # plt.xticks([]), plt.yticks([])
    # plt.colorbar();
    # plt.show()
    #condition1 = high > 40
    # condition2 = ((img / 255) > 0.7)*1
    # conditionb = ((bimg / 255) > 0.7)*1
    # conditioninv = ((invimg / 255) > 0.7)*1
    # condition4 = (img >= bimg)*1
    # print(img.shape)
    # condition1 = (img > invimg)
    # mask1 = (dst * condition2)
    # mask2 = ( mask1*condition1)
    # mask3 = (mask2 *conditionb)
    # mask4 = (mask3 * conditioninv)
    # mask5 = (mask4 * condition4)*1
    # mask *= conditioninv
    # mask *= condition1
    # mask *= condition4
    # mask = minimizing(mask)


    res = 255 * dst
    # plt.imshow(bimg / 255, cmap=c)
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    #  cnts = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)
    # print(cnts)
    # cnts = imutils.grab_contours(cnts)
    #    plt.imshow(cnts * 255, cmap=c)
    #   plt.xticks([]), plt.yticks([])

    plt.show()

    # minimizing(mask)
    # removing to dense canidates
    res1 = 255 * dst  # not minimized full candidate
    if c == 'Reds':
        #mask3 = np.stack((mask2, np.zeros(np.shape(mask), dtype=bool), np.zeros(np.shape(mask), dtype=bool)), axis=2)
        currentframe.AuxRed = res1
        xRed = np.nonzero(res)[0]
        yRed = np.nonzero(res)[1]
        xp = []
        yp = []
        for point in zip(xRed,yRed):
            x =point[0]
            y =point[1]
            if img[x,y] > 178 and img[x,y] > invimg[x,y] and bimg[x,y]> 178 and invimg[x,y] > 178 and img[x,y] >= bimg[x,y]:
                xp.append(x)
                yp.append(y)
        currentframe.xRed = np.array(xp)
        currentframe.yRed = np.array(yp)


    else:
        #mask3 = np.stack((np.zeros(np.shape(mask), dtype=bool), mask2, np.zeros(np.shape(mask), dtype=bool)), axis=2)
        currentframe.AuxGreen = res1
        comparered = (res1 != currentframe.AuxRed)
        xRed = np.nonzero(res)[0]
        yRed = np.nonzero(res)[1]
        xp = []
        yp = []
        for point in zip(xRed, yRed):
            x = point[0]
            y = point[1]
            if (img[x, y] ) > 178 and img[x, y] > invimg[x, y] and (bimg[x, y] ) > 178 and (
                    invimg[x, y] ) > 178 and img[x, y] >= bimg[x, y]:
                xp.append(x)
                yp.append(y)
        currentframe.xGreen = np.array(xp)
        currentframe.yGreen = np.array(yp)
       # print(comparered)
        #currentframe.xGreen = np.nonzero(comparered)[0]
        #currentframe.yGreen = np.nonzero(comparered)[1]

    # org[mask3] = 255
    # plt.imshow(res)
    # plt.colorbar();
    # plt.show()
    # plt.imshow(res1)
    # plt.colorbar();
    # plt.show()
    return org


def add_margin(pil_img, top, right, bottom, left):
    '''for image padding'''
    #   height = pil_img.shape[0]
    #  width =  pil_img.shape[1]
    # new_width = width + right + left
    # new_height = height + top + bottom
    result = np.lib.pad(pil_img, ((left, right), (top, bottom), (0, 0)), 'constant')
    #  result.paste(pil_img, (left, top))
    return result


def CutImagebyP1(img, pointsx, pointsy):
    '''croping the image into images around each point where xpoits is the point in x axis and ypoints in
    y axis'''
    # img_padded= add_margin(img ,81, 81, 81, 81)
    xpoints = pointsx.copy()
    ypoints = pointsy.copy()
    cx1 = xpoints > 80
    cx2 = xpoints < (img.shape[0] - 81)
    xpoints = xpoints[(cx1 * cx2) > 0]
    #print(xpoints)
    cy1 = ypoints > 80
    cy2 = ypoints < (img.shape[1] - 81)
    ypoints = ypoints[(cy1 * cy2) > 0]
    images = []
    for point in zip(pointsx, pointsy):
        # Setting the points for cropped image
        #print(point)
        left = point[1] - 41
        top = point[0] - 41
        right = point[1] + 40
        bottom = point[0] + 40
        # print(img.shape)
        #print(img.shape)
        a = img[top:bottom, left:right, :]
        print(a.shape)
        images.append(img[top:bottom, left:right, :])
    return images, pointsx, pointsy


### functions for part3 calculationg the lacation of a traffic light
def hom(p, focal):
    #     Making the 2d vector into a 3d vector
    return (p[0], p[1], focal)


def dehom(p, focal):
    return (focal * p[0] / p[2], focal * p[1] / p[2])


def foe_and_R_from_EM(EM, focal):
    # extract foe and R from EM
    # t is the translation matrix from em, which is 3 elements from the last row.
    t = EM[:3, 3]
    foe = t / t[2] * focal
    #     print(foe)
    #     print(EM[0,-1]/EM[2,-1]*focal)
    #     print(EM[1,-1]/EM[2,-1]*focal)
    # there is an issue here
    R = EM[:3, :3]
    return foe, R


def calc_epipolar_distance(curr, prev, foe, R, focal):
    m = (foe[1] - curr[1]) / (foe[0] - curr[0])
    n = ((curr[1] * foe[0] - foe[1] * curr[0]) / (foe[0] - curr[0]))
    rot = dehom(R.dot(hom(prev, focal)), focal)
    dist = abs(m * rot[0] + n - rot[1]) / (m ** 2 + 1) ** 0.5
    # debug line to draw the pixels from  a single point to another.
    # dist = dist of rot from line
    return dist


def SFM(curr, prev, foe, R, EM, focal):
    ex = foe[0]
    ey = foe[1]
    # wont run error R doesnt contain tz
    # pass tz to the function here
    tz = EM[2:3, 3]
    # what is deltaxr ???
    rot = dehom(R.dot(hom(prev, focal)), focal)
    xr = rot[0]
    yr = rot[1]
    #     print(f"curr {curr}, ex {ex}, ey {ey}, rot {rot}, tz {tz}")
    Zx = np.dot(tz, ((ex - xr) / (curr[0] - xr)))
    Zy = np.dot(tz, ((ey - yr) / (curr[1] - yr)))
    # Zx = SFM equation based on tracking in x coord
    # Zy = SFM equation based on tracking in y coord
    return Zx, Zy


def calc_TFL_dist(curr_points, prev_points, EM, focal, pp):
    foe, R = foe_and_R_from_EM(EM, focal)
    points_in_world = []
    #print(len(curr_points[0]))
    #print(curr_points)
    points = []
    for idxC in range(len(curr_points[0])):
        possible_matches = []
        color = curr_points[1][idxC]
        #print(curr_points[0][idxC])
        curr = curr_points[0][idxC] - pp
        #         print(f"color {color} curr {curr}")
        for idxP in range(len(prev_points[0])):
            #         if color == prev_points[1][idxP]:
            prev = prev_points[0][idxP] - pp
            epipolar_distance = calc_epipolar_distance(curr, prev, foe, R, focal)
            # if statement for the distance being  a range for number of pixels
            # you will find some of the other lights threshold to get rid some of the noise
            possible_matches.append((epipolar_distance, idxP))
        possible_matches = sorted(possible_matches, key=lambda x: x[0])
        #print(possible_matches)
        if possible_matches[0][0] < 5:
            prev = prev_points[0][possible_matches[0][1]] - pp
            Zx, Zy = SFM(curr, prev, foe, R, EM, focal)
            #         print(f"Zx {Zx}, Zy {Zy}")
            # so print them and choose manually
            #         ex,ey = foe[0],foe[1]
            #         Z = Zy
            #         if abs(curr[0]-ex) > (curr[1]-ey) :
            #             Z = Zx
            #      Z = max(Zx,Zy)
            Z = (abs(EM[0, 3] * Zx) + abs(EM[1, 3] * Zy)) / (abs(EM[0, 3]) + abs(EM[1, 3]))
            #         Z = min(Zx,Zy)
            #         print(f"Z is {Z}")
            # difference between foe and current pixel, the largest the more you trurst the calculation, there is more movement
            #     print zx and zy and choose which is better in a way.
            # Z = combine zx and zy to your undestanding
            #       P = calc 3D point from curr and Z
            px = curr[0] * (Z / focal)
            py = curr[1] * (Z / focal)
            P = np.concatenate((px, py, Z))
            points_in_world.append(P)
            points.append([curr_points[0][idxC][0], curr_points[0][idxC][1]])
    return points_in_world, points
