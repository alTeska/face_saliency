# development file, later on itti&koch class
import math
import numpy as np
import scipy.signal as signal
from scipy import ndimage as nd
import matplotlib.image as mpimg
from skimage import img_as_float64

from .utils import *
from .itti_koch_features import *


class IttiKoch():
    '''
    Itti and Koch main model class.
    Inputs upon init: path to the image, changes to setup dict
    '''
    def __init__(self, input_params=None, verbose=True):
        super().__init__()

        self.verbose = verbose

        # dictionary with default params
        self.params = {
            "min_mapwidth": 64,
            "num_center_scales": 3,
            "outer_sigma": [3,6],
            "inner_sigma": [1,1],
            "topdown_weights": [1,1,1],
            "gabor_theta": 4,
            "gabor_sigma": 4,
            "gabor_frequency": 0.1,
            "gabor_phase": False,
            "gaussian_blur": 2,
            "fraction_centerbias": 2
        }


        # update the parameters with the input
        if input_params:
            for key in input_params:
                self.params[key] = input_params[key]

            pass


    def make_center_scales(self, img, scalars):
        '''
        Gaussian pyramid creation: based on image and scalars
        returns a list of downsampled images with lenght according to inp scalars
        '''
        # determine size of center scales
        self.params["mapsize"] = (round(img.shape[0] * (self.params["min_mapwidth"] / img.shape[1])), self.params["min_mapwidth"])

        return downsample_image(img, self.params["mapsize"][0], self.params["min_mapwidth"], scalars)


    def make_feature_maps(self, imgs, feature_func, *args):
        '''
        Given an image and a function for the feature, calculates the convolved
        feature maps.
        Returns: convolution maps
        '''
        feat_maps = feature_func(imgs, *args)

        return convolve_receptive_field(feat_maps, self.params["inner_sigma"], self.params["outer_sigma"])


    def make_conspicuity_maps(self, img_scales, feature_func, *args):
        '''
        Function creates conspicuity map based on given image scales for given feature
        '''
        conspicuity_maps = []
        for img in img_scales:
            feature_maps = self.make_feature_maps([img], feature_func, *args)
            conspicuity_map = compute_conspicuity_map(feature_maps, self.params["mapsize"], resize_map=False)
            # append the normalized conspicuity map
            conspicuity_maps.append(conspicuity_map)

        # compute weighted sum across images scales, do resizing to deal with different image sizes
        conspicuity_map = compute_conspicuity_map(conspicuity_maps, self.params["mapsize"], resize_map=True)

        return conspicuity_map

    def run(self, img, keys = ["intensity", "color", "orientation"], verbose = False):
        '''
        Given an image returns its saliency and the single saliency maps in the order of feature  input
        '''
        img = img_as_float64(img) # convert to doubles if image is uint8

        # compute spatial scales
        if self.verbose:
            print("Computing {} image scales".format(self.params["num_center_scales"]))
        img_scales = self.make_center_scales(img, np.arange(self.params["num_center_scales"])+1)

        if self.verbose:
            print("Creating Gabor kernels for orientation.")
        gabor_kernels = create_gabor_kernels(theta = self.params["gabor_theta"],
                                            sigma = self.params["gabor_sigma"],
                                            frequency = self.params["gabor_frequency"],
                                            phase = self.params["gabor_phase"])

        # compute conspicuity_map for each channel
        saliency_maps = []

        # iterate over features to compute (keys)
        for key in keys:
            if self.verbose:
                print("Computing saliency maps for {}.".format(key))

            # get corresponding function
            curr_func = globals()["compute_"+key]

            # compute conspicuity map
            if (key == "orientation"):
                saliency_maps.append(self.make_conspicuity_maps(img_scales, curr_func, gabor_kernels))
            else:
                saliency_maps.append(self.make_conspicuity_maps(img_scales, curr_func))

        # sum & normalize across channels
        wj = self.params["topdown_weights"]

        saliency = np.zeros(self.params["mapsize"])
        for i in np.arange(len(saliency_maps)):
            saliency = saliency + wj[i]*saliency_maps[i]

        # blur the image
        saliency = nd.gaussian_filter(saliency, self.params["gaussian_blur"])

        # normalize with the maximum of the map
        saliency = saliency / np.max(saliency)

        # introduce center bias
        saliency = saliency * center_bias(lambda x, y: gaussian2D(x, y, min(np.shape(saliency)) / self.params["fraction_centerbias"]), np.shape(saliency))

        # return saliency
        return saliency, saliency_maps
