from scipy.misc import imsave
import Utils
import numpy as np
import os
import matplotlib.pyplot as plt

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


class TFL_Man(Singleton):
    prevframe = {}
    currentframe = {}

    def __init__(self, frameID, framepath):
        super().__init__()
        self.currentframe['FramId'] = frameID
        self.currentframe['FramePath'] = framepath

    def Setframe(self, frameID, framepath):
        self.prevframe = self.currentframe.copy()
        self.currentframe['FramId'] = frameID
        self.currentframe['FramePath'] = framepath

    def Part1(self, output1_dir):
        image = Utils.Read_image(self.currentframe['FramePath'])

        drawed = np.copy(image)
        name = self.currentframe['FramePath'].split('\\')[-1]
        for i, c in enumerate(('Reds', 'Greens')):
            img = np.copy(image)
            Utils.findcandidate(self.currentframe, image[:, :, i], c, drawed)
        dir_name = os.path.join(output1_dir, name)
        print(self.currentframe['FramePath'])
        plt.figure()
        plt.imshow(drawed/ 255)
        plt.savefig(dir_name)
        plt.close()

