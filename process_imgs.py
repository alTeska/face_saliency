import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from saliency_model.itty_koch import IttiKoch

plt.rcParams['image.cmap'] = 'gray'

# Load The Image
path = './imgs/balloons.png'
img = mpimg.imread(path)

IK = IttiKoch(path)
saliency_maps, saliency_map, saliency_maps_o, saliency_map_o, saliency_maps_c, saliency_map_c = IK.run()


fig, ax = plt.subplots(1, 4, figsize=(15,5))
fig.suptitle('intensity saliency')
ax[0].imshow(saliency_maps[0])
ax[1].imshow(saliency_maps[1])
ax[2].imshow(saliency_maps[2]);
ax[3].imshow(saliency_map);

fig, ax = plt.subplots(1, 4, figsize=(15, 5))
fig.suptitle('orientation saliency')
ax[0].imshow(saliency_maps_o[0])
ax[1].imshow(saliency_maps_o[1])
ax[2].imshow(saliency_maps_o[2]);
ax[3].imshow(saliency_map_o);

fig, ax = plt.subplots(1, 4, figsize=(15, 5))
fig.suptitle('color saliency')
ax[0].imshow(saliency_maps_c[0])
ax[1].imshow(saliency_maps_c[1])
ax[2].imshow(saliency_maps_c[2]);
ax[3].imshow(saliency_map_c);


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(saliency_map)
ax[1].imshow(saliency_map_o)
ax[2].imshow(saliency_map_c)
ax[0].set_title('intensity')
ax[1].set_title('orientation')
ax[2].set_title('color')

plt.show()
