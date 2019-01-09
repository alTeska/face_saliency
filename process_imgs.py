import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from saliency_model.itty_koch import IttiKoch

plt.rcParams['image.cmap'] = 'gray'

# Load The Image
path = './imgs/balloons.png'
img = mpimg.imread(path)

IK = IttiKoch(path)
img_list, img_int = IK.run()


fig, ax = plt.subplots(1, 3, figsize=(15, 15))
ax[0].imshow(img_list[0])
ax[1].imshow(img_list[1])
ax[2].imshow(img_list[2]);

fig, ax = plt.subplots(1, 3, figsize=(15, 15))
ax[0].imshow(img_int[0])
ax[1].imshow(img_int[1])
ax[2].imshow(img_int[2]);

plt.show()
