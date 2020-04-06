from scipy.misc import imsave
import Utils
import numpy as np
import os
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from imageio import imread
import scipy.misc


class Singleton:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Singleton.__instance is None:
            Singleton()
        return Singleton.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Singleton.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Singleton.__instance = self


class Container:
    def __init__(self, FramId, FramePath):
        self.FrameId = FramId
        self.FramePath = FramePath
        self.AuxRed = np.array([])
        self.xRed = np.array([])
        self.yRed = np.array([])
        self.AuxGreen = np.array([])
        self.xGreen = np.array([])
        self.yGreen = np.array([])
        self.Part2res = ()  # this contains the trafic light points detected as TF according to the NN and in a format ready for part 2
        self.locations = []
        self.tf_3Dlocations = []


class TFL_Man(Singleton):
    # prevframe = {}
    # currentframe = {}

    def __init__(self, frameID, framepath):
        super().__init__()
        # self.currentframe = frameID
        # self.currentframe['FramePath'] = framepath
        self.currentframe = Container(frameID, framepath)
        self.prevframe = Container(frameID, framepath)

    def Setframe(self, frameID, framepath):
        self.prevframe = self.currentframe
        # print(self.prevframe.xRed)
        # print(self.prevframe.xGreen)
        self.currentframe = Container(frameID, framepath)

        # self.prevframe = self.currentframe.copy()
        # self.currentframe['FramId'] = frameID
        # self.currentframe['FramePath'] = framepath

    def Part1(self, output1_dir):
        image = Utils.Read_image(self.currentframe.FramePath)
        # im_pil = Image.open(self.currentframe.FramePath)
        drawed = np.copy(image)
        name = self.currentframe.FramePath.split('\\')[-1]
        for i, c in enumerate(('Reds', 'Greens')):
            img = np.copy(image)
            k = (i+1)%2

            Utils.findcandidate(self.currentframe, image[:, :, i],image[:,:,k],image[:,:,2], c, drawed)
        dir_name = os.path.join(output1_dir, name)
        print(self.currentframe.FramePath)
        plt.figure()
        plt.imshow(drawed / 255)
        plt.savefig(dir_name)
        plt.close()

    def Part2(self):
        ''' returns or extracts from the candidates who is among them detected as trafic light by the neural network model which is
        uploaded from the export pkl'''
        colors = []
        true_candidate = []
        image = Utils.Read_image(self.currentframe.FramePath)
        cutted_imgs_dir = r'C:\Users\RENT\Desktop\Mobilye project\dusseldorf_000049\cuttedimage_part2'

        print(type(image))
        for idxc, c in enumerate(('Reds', 'Greens')):
            if idxc == 0:

                images, xpoints, ypoints = Utils.CutImagebyP1(image, self.currentframe.xRed.copy(),
                                            self.currentframe.yRed.copy())  # this returns the images cropped around each candidate
            else:
                images, xpoints, ypoints = Utils.CutImagebyP1(image, self.currentframe.xGreen.copy(),
                                            self.currentframe.yGreen.copy())  # this returns the images cropped around each candidate
            learn2 = load_learner('./')

            for i, img in enumerate(images):
                img_path = os.path.join(cutted_imgs_dir, str(self.currentframe.FrameId) + str(i) + '.jpg')
                scipy.misc.imsave(img_path, img)
                fastai_img = open_image(img_path)
                pred_class, pred_idx, outputs = learn2.predict(fastai_img)

                print(str(pred_class))
                cat = str(pred_class)
                # print(cat)
                # print(pred_class.data.eval())
                if (cat == 'Yes'):
                    plt.imshow(img / 255)
                    plt.show()
                    print(i)
                    colors.append(c)
                    true_candidate.append([ypoints[i], xpoints[i]])

            if true_candidate:
                self.currentframe.Part2res = (np.array(true_candidate), colors)

    def Part3(self, dir_name):
        ''' return the the estimated locations of each of the pointes that part 2 outputed'''
        if self.currentframe.Part2res and self.prevframe.Part2res:
            img_curr = Utils.Read_image(self.currentframe.FramePath)
            img_prev = Utils.Read_image(self.prevframe.FramePath)
            #    with open(self.prevframe.FramePath, 'rb') as imgfile:
            #       img_prev = imread(imgfile)
            #  with open(self.currentframe.FramePath, 'rb') as imgfile:
            #     img_curr = imread(imgfile)
            with open('dusseldorf_000049/dusseldorf_000049.pkl', 'rb') as pklfile:
                data = pickle.load(pklfile, fix_imports=True, encoding='latin1')
            focal_length = data['flx']
            pp = data['principle_point']
            #print('egomotion_' + str(self.prevframe.FrameId) + '-' + str(self.currentframe.FrameId))
            EM18_19 = data['egomotion_' + str(self.prevframe.FrameId) + '-' + str(self.currentframe.FrameId)]
            self.currentframe.tf_3Dlocations , points = Utils.calc_TFL_dist(self.currentframe.Part2res, self.prevframe.Part2res,EM18_19, focal_length, pp)
            #points_18 = data['points_18']
            #points_19 = data['points_19']
            #EM18_19 = data['egomotion_18-19']
            #.currentframe.tf_3Dlocations = Utils.calc_TFL_dist(points_19, points_18, EM18_19, focal_length, pp)
            #  EM18_19, focal_length, pp)


            # for visualation
            #points_0 = data['points_00']
            #points_1 = data['points_01']
            plt.figure('frame prev')
            plt.imshow(img_prev/255)
            plt.scatter(self.currentframe.Part2res[0][:, 0], self.currentframe.Part2res[0][:, 1])
            plt.figure('frame curr')
            plt.imshow(img_curr/255)
            plt.scatter(self.prevframe.Part2res[0][:, 0], self.prevframe.Part2res[0][:, 1])
            plt.show()
            style = dict(size=15, color='yellow')
            plt.figure(figsize=(20, 20))
            plt.imshow(img_curr / 255)

            x = [x[0] for x in points]
            y = [x[1] for x in points]
            plt.scatter(x, y)
            for i in range(len(self.currentframe.tf_3Dlocations)):
                plt.annotate(f"{int(self.currentframe.tf_3Dlocations[i][2])}m",
                             (points[i][0], points[i][1]), **style)
            plt.show()
            #plt.savefig(path)
            print(self.currentframe.tf_3Dlocations)

