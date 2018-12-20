import math
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

def gaussian2D(x, y, sigma):
    return (1.0/(1*math.pi*(sigma**2)))*math.exp(-(1.0/(2*(sigma**2)))*(x**2 + y**2))

def mexicanHat(x,y,sigma1,sigma2):
    return gaussian2D(x,y,sigma1) - gaussian2D(x,y,sigma2)

def receptiveFieldMatrix(func):
    """make matrix from function"""
    h = 30
    g = np.zeros((h,h))
    for xi in range(0,h):
        for yi in range(0,h):
            x = xi-h/2
            y = yi-h/2
            g[xi, yi] = func(x,y);
    return g

def plotFilter(fun):
    g = receptiveFieldMatrix(fun)
    plt.imshow(g, cmap=cm.Greys_r)



# Load The Image
img = mpimg.imread('./imgs/balloons.png')
img2 = cv2.imread('./imgs/balloons.png')

print(np.shape(img))
print(np.shape(img2))
img = img[:,:,0]

# convert to double if image is uint8
img_hat = signal.convolve(img, receptiveFieldMatrix(lambda x,y: mexicanHat(x,y,2,3)), mode='same')

# determine size and number of Center scales


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
