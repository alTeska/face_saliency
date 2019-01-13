# development file, later on itti&koch class
import math
import numpy as np
import scipy.signal as signal
import matplotlib.image as mpimg
from skimage import img_as_float64

from .utils import *
from .itti_koch_features import *


class IttiKoch():
    '''
    Itti and Koch main model class.
    Inputs upon init: path to the image, changes to setup dict
    '''
    def __init__(self, img):
        super().__init__()

        self.img = img_as_float64(img)      # convert to doubles if image is uint8
        self.mapwidth = 64
        self.outer_sigma = [3,6]
        self.inner_sigma = [1,1]

        self.weights = [1,1,1]

        pass

    def make_center_scales(self, img, scalars):
        '''
        Gaussian pyramid creation: based on image and scalars creates a list of
        downsampled images with lenght according to inp scalars
        '''
        # determine size of center scales
        self.mapsize = (round(img.shape[0] * (self.mapwidth / img.shape[1])), self.mapwidth)
        img_scales = downsample_image(img, self.mapsize[0], self.mapwidth, scalars)

        return img_scales


    def make_feature_maps(self, imgs, feature_func, *args):
        '''
        Given an image and a function for the feature, calculates the convolved
        feature maps.
        '''
        feat_maps = feature_func(imgs, *args)
        conv_maps = convolve_receptive_field(feat_maps, self.inner_sigma, self.outer_sigma)

        return conv_maps

    def make_conspicuity_maps(self, img_scales, feature_func, *args):
        '''
        Function creates conspicuity map based on given image scales for given feature
        '''
        conspicuity_maps = []
        for img in img_scales:
            feature_maps = self.make_feature_maps([img], feature_func, *args)
            conspicuity_map = compute_conspicuity_map(feature_maps, self.mapsize)
            conspicuity_maps.append(conspicuity_map)

        # sum & normalize across scales
        conspicuity_map = compute_conspicuity_map(conspicuity_maps, self.mapsize)
        return conspicuity_map


    def run(self):
        img = self.img

        wj = [.3, .3, .3]   # linear combination of weights
        gabor_kernels = create_gabor_kernels()

        # compute spatial scales
        img_scales = self.make_center_scales(img, [1, 2, 3])

        # compute conspicuity_map for each channel
        intensity = self.make_conspicuity_maps(img_scales, compute_intensity)
        color = self.make_conspicuity_maps(img_scales, compute_color)
        orientation = self.make_conspicuity_maps(img_scales, compute_orientation, gabor_kernels)

        # sum & normalize across channels
        saliency = wj[0]*intensity + wj[1]*color + wj[2]*orientation

        # return saliency
        return saliency, intensity, color, orientation
