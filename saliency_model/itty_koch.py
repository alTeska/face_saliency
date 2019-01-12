# development file, later on itti&koch class
import math
import numpy as np
import scipy.signal as signal
import matplotlib.image as mpimg

from .utils import *
from .itti_koch_features import *


class IttiKoch():
    '''
    Itti and Koch main model class.
    Inputs upon init: path to the image, changes to setup dict
    '''
    def __init__(self, path):
        super().__init__()

        # Load The Image
        self.img = mpimg.imread(path)

        self.mapwidth = 64
        self.outer_sigma = [3,6]
        self.inner_sigma = [1,1]

        self.weights = [1,1,1]

        pass

    def make_feature_map(self, imgs, feature_func, *args):
        '''given a function for the feature, calculates the convolved feature map'''
        feat_maps = feature_func(imgs, *args)
        conv_maps = convolve_receptive_field(feat_maps, self.inner_sigma, self.outer_sigma)

        return conv_maps

    def run(self):
        img = self.img

        # TODO convert to double if image is uint8

        # create gabor kernels
        gabor_kernels = create_gabor_kernels()

        # determine size and number of Center scales
        self.mapsize = (round(img.shape[0] * (self.mapwidth / img.shape[1])), self.mapwidth)
        scalars = [1, 2, 3]

        img_scales = downsample_image(img, self.mapsize[0], self.mapwidth, scalars)

        num_chan = 3 # TODO: to normalize the channels


        saliency_maps = []
        saliency_maps_o = []
        saliency_maps_c = []


        for img in img_scales:
            # calculate chosen feature
            intensity_maps = self.make_feature_map([img], compute_intensity)
            color_maps = self.make_feature_map([img], compute_color)
            orientation_maps = self.make_feature_map([img], compute_orientation, gabor_kernels)

            # normalize
            saliency_maps.append(compute_saliency_map(intensity_maps, self.mapsize))
            saliency_maps_c.append(compute_saliency_map(color_maps, self.mapsize))
            saliency_maps_o.append(compute_saliency_map(orientation_maps, self.mapsize))


        # sum across scales & normalize
        saliency_map = compute_saliency_map(saliency_maps, self.mapsize)
        saliency_map_c = compute_saliency_map(saliency_maps_c, self.mapsize)
        saliency_map_o = compute_saliency_map(saliency_maps_o, self.mapsize)


        # TODO sum together maps across channels
        saliency = saliency_map/3 + saliency_map_c/3 + saliency_map_o/3

        return saliency, saliency_map, saliency_map_c, saliency_map_o
        # return saliency, saliency_map
