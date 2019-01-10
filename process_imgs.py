import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from saliency_model.itty_koch import IttiKoch

plt.rcParams['image.cmap'] = 'gray'

# Load The Image
path = './imgs/balloons.png'
img = mpimg.imread(path)

IK = IttiKoch(path)
saliency_maps, saliency_map = IK.run()


fig, ax = plt.subplots(1, 4, figsize=(15, 15))
ax[0].imshow(saliency_maps[0])
ax[1].imshow(saliency_maps[1])
ax[2].imshow(saliency_maps[2]);
ax[3].imshow(saliency_map);


plt.show()
