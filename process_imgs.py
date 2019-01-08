import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from saliency_model.itty_koch import IttiKoch

# Load The Image
path = './imgs/balloons.png'
img = mpimg.imread(path)

IK = IttiKoch(path)
img_hat, img_list = IK.run()

fig, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].imshow(img)
ax[1].imshow(img_hat)

fig, ax = plt.subplots(1, 3, figsize=(15, 15))
ax[0].imshow(img_list[0])
ax[1].imshow(img_list[1])
ax[2].imshow(img_list[2]);

plt.show()
