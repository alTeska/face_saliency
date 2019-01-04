import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as transform


def gaussian2D(x, y, sigma):
    return (1.0/(1*math.pi*(sigma**2)))*math.exp(-(1.0/(2*(sigma**2)))*(x**2 + y**2))


def mexicanHat(x, y, sigma1, sigma2):
    return gaussian2D(x, y, sigma1) - gaussian2D(x, y, sigma2)


def receptiveFieldMatrix(func):
    """make matrix from function"""
    h = 30
    g = np.zeros((h, h))
    for xi in range(0, h):
        for yi in range(0,h):
            x = xi-h/2
            y = yi-h/2
            g[xi, yi] = func(x, y)
    return g


def plotFilter(fun):
    g = receptiveFieldMatrix(fun)
    #plt.imshow(g, cmap=plt.cm.Greys_r


def downsample_image(image, min_height, min_width, scaling_factors):
    '''Downsamples the image to a smaller map size, while keeping the third dimension as it is.
    Returns a list of downsampled images with a length according to input scalars.
    Scaling factors have to be integers!'''
    img_maps = []
    for scalar in scaling_factors:
        img_maps.append(transform.resize(image, ((min_height * scalar), (min_width * scalar)), mode='constant', anti_aliasing=True))
    return img_maps
