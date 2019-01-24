# Basic features implemented by the Itti-Koch Algorithm
import skimage
import numpy as np
<<<<<<< 5408f8f14e524e53c8537d7891f3cf2d1cd4876b
# TODO change back!
from utils import convolve_kernels
=======
import face_recognition
from .utils import convolve_kernels, fit_gauss_to_rectangle
>>>>>>> face added to IK as a separate flag


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

    return np.squeeze(img_col)


def compute_orientation(img_list, gabor_kernels):
    """
    Computes the mean over the RGB values of the image to get intensity value.
    :param img_list: image in different scales
    :return: list of orientation values for each image
    """

    intensity = (compute_intensity(img_list))
    orientation = convolve_kernels(np.squeeze(intensity), gabor_kernels)

    return orientation


def apply_face_saliency(image, face_locations):
    '''
    Function creates a saliency map for face recognition feature, based on pre-detected face locations
    '''
    # TODO: add blur ?

    face_saliency = np.zeros_like(image[:,:,0], dtype='float64')

    for face_location in face_locations:
        # get all the face location and fit the gaussian into it
        top, right, bottom, left = face_location
        center = fit_gauss_to_rectangle(top, bottom, right, left)

        # overwrtie the face are with gaussian
        face_saliency[top:bottom, left:right] = center

    # face_saliency = nd.gaussian_filter(face_saliency, blur)
    return face_saliency


def compute_faces(img):
    '''
    Computes saliency map for faces
    '''


    img1 = skimage.img_as_ubyte(img[:,:,0:3])

    face_locations = face_recognition.face_locations(img1)
    saliency = apply_face_saliency(img, face_locations)


    # print(img1.dtype, '\n')
    # print(img.dtype, '\n')
    # print(np.shape(img1), '\n')
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(ncols=2, nrows=1)
    # ax[0].imshow(img1[:,:,0])
    # ax[1].imshow(img1[:,:,1])
    # plt.show()
    
    return saliency
