# development file, later on itti&koch class
import math
import numpy as np
import scipy.signal as signal
import matplotlib.image as mpimg

from .utils import receptive_field_matrix, mexican_hat, downsample_image
from itti_koch_features import *


class IttiKoch():
    '''
    Itti and Koch main model class.
    Inputs upon init: path to the image, changes to setup dict
    '''
    def __init__(self, path):
        super().__init__()

        # Load The Image
        self.img = mpimg.imread(path)
        pass

    def run(self):
        img = self.img[:,:,0]

        # convert to double if image is uint8
        img_hat = signal.convolve(img, receptive_field_matrix(lambda x,y: mexican_hat(x,y,2,3)), mode='same')

        # determine size and number of Center scales
        mapwidth = 64
        mapheight = round(img.shape[0] * (mapwidth / img.shape[1]))
        scalars = [1, 2, 3]

        img_list = downsample_image(img, mapheight, mapwidth, scalars)

        # split to channels & compute salience in each
        img_int = compute_intensity(img_list)

        # each channel apply the center surround

        # normalize channels


        # sum together maps across channels
        # fig, ax = plt.subplot(1, 3)
        #
        # ax[0] = plt.imshow(img[:,:,0])


        return img_hat, img_list
