import math
import cv2
import face_recognition
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage as nd
from saliency_model.utils import fit_gauss_to_rectangle



def apply_face_saliency(image, face_locations, blur=2):
    '''
    Function creates a saliency map for face recognition feature, based on pre-detected face locations
    '''
    # TODO: add blur depending on the size of the picture

    face_saliency = np.zeros_like(image[:,:,0], dtype='float64')

    for face_location in face_locations:
        # get all the face location and fit the gaussian into it
        top, right, bottom, left = face_location
        center = fit_gauss_to_rectangle(top, bottom, right, left)

        # overwrtie the face are with gaussian
        face_saliency[top:bottom, left:right] = center

    # add blur
    saliency = nd.gaussian_filter(face_saliency, blur)
    return saliency



image = face_recognition.load_image_file("imgs/group.jpg")
# image = face_recognition.load_image_file("imgs/biden.jpg")
# image = face_recognition.load_image_file("/imgs/baby1.png")
# image = face_recognition.load_image_file("/imgs/1.jpg")

face_locations = face_recognition.face_locations(image)
saliency = apply_face_saliency(image, face_locations)

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 10))
ax[0].imshow(image);
ax[1].imshow(saliency, cmap='gray', vmin=0, vmax=1);

plt.show()
