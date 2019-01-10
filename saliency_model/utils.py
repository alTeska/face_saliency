import math
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import skimage.transform as transform
import warnings

from skimage.feature import peak_local_max
from skimage.transform import resize


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


def plot_filter(fun):
    '''TODO: decide about this function '''
    g = receptiveFieldMatrix(fun)
    #plt.imshow(g, cmap=plt.cm.Greys_r


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
            img_conv = signal.convolve2d(img, rf1, mode='same')  # convolve image with narrow gaussian
            convolved = signal.convolve2d(img, rf2, mode='same') # convolve image with wide gaussian
            
            # substract one conv. img with the other
            # corresponds to creating the mexican hat function
            conv = (img_conv - convolved)**2
            
#             img_hat.append(conv)                     # save as 2D list
#             output_list.append(img_hat)

            # list containing convolved versions of all images
            output_list.append(conv) # save as 1D list

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
        
    for x,y in coordinates:
        peak_avg = peak_avg + sal_map[x][y]
    
    if (peak_num != 0):
        peak_avg = peak_avg/peak_num
    
    return peak_avg, peak_num


def compute_saliency_map(convolution_maps, mapsize):
    weights = []
    saliency_map = np.zeros(mapsize)

    for i,m in enumerate(convolution_maps):
    
        # normalize the maps to have values between 0 and 1
        m = normalize(m)
    
        # get local maxima of the map
        peak_avg, peak_num = get_local_maxima(m, min_distance=1)
        weights.append(get_weight_map(peak_avg, peak_num))
    
        temp = resize(m, mapsize, mode='constant', anti_aliasing=True)
        saliency_map += weights[i] * temp
    
    return saliency_map