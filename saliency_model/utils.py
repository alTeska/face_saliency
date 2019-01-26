import math
import warnings
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import skimage.transform as transform

from skimage.feature import peak_local_max
from skimage.transform import resize
from skimage.filters import gabor_kernel
from scipy import ndimage as nd


def gaussian2D(x, y, sigma):
    '''create 2D Gaussian distribution'''
    return (1.0/(2*math.pi*(sigma**2)))*math.exp(-(1.0/(2*(sigma**2)))*(x**2 + y**2))


def mexican_hat(x, y, sigma1, sigma2):
    ''' @ gausain mixture(difference) creates mexican hat like shape '''
    return gaussian2D(x, y, sigma1) - gaussian2D(x, y, sigma2)


def receptive_field_matrix(func):
    """make matrix from function"""
    h = 30
    g = np.zeros((h, h))
    for xi in range(0, h):
        for yi in range(0,h):
            x = xi-h/2
            y = yi-h/2
            g[xi, yi] = func(x, y)
    return g

# TODO: basically a copy of the receptive field function - combine them!
def center_bias(func, mapsize):
    """make matrix from function"""
    g = np.zeros(mapsize)
    for xi in range(0, mapsize[0]):
        for yi in range(0,mapsize[1]):
            x = xi-mapsize[0]/2
            y = yi-mapsize[1]/2
            g[xi, yi] = func(x, y)
    # normalize to a height of one
    g = g / np.max(g)
    return g
    

def fit_gauss_to_rectangle(top, bottom, right, left):
    '''
    input: top, bottom, right and left coordinates of the rectangle
    Returns 2D distribution for saliency fit into the rectangle
    '''
    sigma = (bottom - top)/4
    center = center_bias(lambda x, y: gaussian2D(x, y, sigma=sigma), (bottom-top, right-left))

    return center


def create_gabor_kernels(theta=4, sigma=4, frequency=0.1, phase=False):
    '''
    Function returns multiple Gabor Filters kernels. Theta defines how many
    different angles will be taken into accout: 180 / theta
    (example: theta=4 -> angles [0, 45, 90, 135])
    '''
    kernels = []

    for theta in range(int(theta)):
        theta = theta / 4. * np.pi
        kernel = np.real(gabor_kernel(frequency, theta=theta,
                                      sigma_x=sigma, sigma_y=sigma,
                                      offset=phase))
        kernels.append(kernel)

    return kernels


def convolve_kernels(image, kernels):
    '''compute features for multiple kernels: convolve with image'''
    feats = []

    for kernel in kernels:
        filtered = nd.convolve(image, kernel)
        feats.append(filtered)

    return feats


def downsample_image(image, min_height, min_width, scaling_factors):
    '''
    Downsamples the image to a smaller map size, while keeping the third dimension as it is.
    Returns a list of downsampled images with a length according to input scalars.
    Scaling factors have to be integers!
    '''
    img_maps = []

    for scalar in scaling_factors:
        img_maps.append(transform.resize(image, ((min_height * scalar),
                                         (min_width * scalar)), mode='constant',
                                         anti_aliasing=True))

    return img_maps


def convolve_receptive_field(input_list, sigma1, sigma2):
    """
    convolves all images of the input list with the sigma-combinations;
    returns a nested list of convolved images, where the first index is associated with the image,
    and the second index is associated with the sigmas (kernel-size)
    """
    if (len(sigma1) != len(sigma2)):
        warnings.warn("Amount of sigmas for inner and outer Gaussian are not the same!")

    # outer loop over the feature images provided
    output_list = []
    for img in input_list:

        # inner loop over the sigma-combinations
        img_hat = []
        for s1, s2 in zip(sigma1, sigma2):
            # create receptive field and convolve with image
            rf1 = receptive_field_matrix(lambda x, y: gaussian2D(x, y, s1))
            rf2 = receptive_field_matrix(lambda x, y: gaussian2D(x, y, s2))

            # list containing different kernel sizes
            img_conv = signal.convolve2d(img, rf1, mode='same')   # convolve image with narrow gaussian
            convolved = signal.convolve2d(img, rf2, mode='same')  # convolve image with wide gaussian

            # substract one conv. img with the other
            # corresponds to creating the mexican hat function
            conv = (img_conv - convolved)**2

            # list containing convolved versions of all images
            output_list.append(conv)

    return output_list


def normalize(X):
    '''
    normalizes the given array to numbers scaled between 0 and 1
    '''
    return (X - X.min()) / (X.max() - X.min())


def get_weight_map(peak_avg, peak_num):
    '''
    based on number of peaks and their average return weight of the map used for
    normalization, measure of peakiness
    '''

    if peak_num <= 1:
        return 1
    else:
        return (1 - peak_avg)**2


def get_local_maxima(sal_map, min_distance=1, threshold_abs=0.1):
    '''
    Input image, find local minima with min_distance window between points.
    Return average of local minimas and their number
    '''

    coordinates = peak_local_max(sal_map, min_distance=min_distance, threshold_abs=threshold_abs)

    peak_avg = 0
    peak_num = np.shape(coordinates)[0]

    for x, y in coordinates:
        peak_avg = peak_avg + sal_map[x][y]

    if (peak_num != 0):
        peak_avg = peak_avg/peak_num

    return peak_avg, peak_num


def compute_conspicuity_map(convolution_maps, mapsize, resize_map = True):
    '''
    Function computes conspicuity maps based on convolution_maps and expected mapsize.
    Maps get scaled between 0-1, then local minima are found and averaged for additional scaling.
    Afterwards maps get normalized / scaled between 0 and 1 again.
    '''

    weights = []
    sum_weights = 0
    conspicuity_map = np.zeros_like(convolution_maps[0])

    for i, m in enumerate(convolution_maps):

        # normalize the maps to have values between 0 and 1
        m = normalize(m)

        # get local maxima of the map
        peak_avg, peak_num = get_local_maxima(m, min_distance=1)
        weights.append(get_weight_map(peak_avg, peak_num))

        if (resize_map):
            m = resize(m, mapsize, mode='constant', anti_aliasing=True)

        conspicuity_map += weights[i] * m

    conspicuity_map = normalize(conspicuity_map)

    return conspicuity_map
