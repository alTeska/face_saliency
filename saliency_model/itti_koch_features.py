# Basic features implemented by the Itti-Koch Algorithm

import numpy as np


def compute_intensity(img_list):
    """
    Computes the mean over the RGB values of the image to get intensity value.
    :param img_list: image in different scales
    :return: list of intensity values for each image
    """

    img_avg = []

    for img in img_list:
        img_avg.append(np.mean(img, axis=2))

    return img_avg



def compute_color(img_list):
    '''
    Separates the channels by color, calculates the luminescence as their mean,
    :param img_list: image in different scales
    :return: rg, by colors
    '''
    
    img_col = []

    for img in img_list:

        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]

        lum = np.mean([r,g,b])

        by = np.divide((b - np.minimum(r, g)), lum)  # (B-Y)
        rg = np.divide((r - g), lum)                 # (R-G)
        img_col.append([rg, by])

    return img_col
