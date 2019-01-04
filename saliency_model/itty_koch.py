# development file, later on itti&koch class 

import math
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import saliency_model.utils

# Load The Image
img = mpimg.imread('./imgs/balloons.png')
img2 = cv2.imread('./imgs/balloons.png')

print(np.shape(img))
print(np.shape(img2))
img = img[:,:,0]

# convert to double if image is uint8
img_hat = signal.convolve(img, receptiveFieldMatrix(lambda x,y: mexicanHat(x,y,2,3)), mode='same')

# determine size and number of Center scales
mapwidth = 64
mapheight = round(img.shape[0] * (mapwidth / img.shape[1]))
scalars = [1, 2, 3]

img_list = downsample_image(img, mapheight, mapwidth, scalars)

# split to channels & compute salience in each


# each channel apply the center surround


# normalize channels


# sum together maps across channels
fig, ax = plt.subplot(nrow=1, ncol=3)

ax[0] = plt.imshow(img[:,:,0])

plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(img_hat)

plt.show()
