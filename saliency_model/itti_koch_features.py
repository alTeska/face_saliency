# Basic features implemented by the Itti-Koch Algorithm

import numpy as np


def compute_intensity(img_list):
    """
    Computes the mean over the RGB values of the image to get intensity value.
    :param img_list: image in different scales
    :return: list of intensity values for each image
    """
    img_avg = []

    for i in img_list:
        img_avg.append(np.mean(i, axis=2))

    return img_avg
