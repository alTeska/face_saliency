import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from saliency_model.itty_koch import IttiKoch

plt.rcParams['image.cmap'] = 'gray'

# Load The Image
path = './imgs/balloons.png'
img = mpimg.imread(path)

IK = IttiKoch(path)
saliency, saliency_map_i, saliency_map_c, saliency_map_o = IK.run()

fig, ax = plt.subplots(1, 4, figsize=(15, 5))
ax[0].imshow(saliency_map_i)
ax[1].imshow(saliency_map_o)
ax[2].imshow(saliency_map_c)
ax[3].imshow(saliency)
ax[0].set_title('intensity')
ax[1].set_title('orientation')
ax[2].set_title('color')
ax[3].set_title('ICO')

plt.show()
