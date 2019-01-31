import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from saliency_model.itti_koch import IttiKoch

plt.rcParams['image.cmap'] = 'gray'

# Load The Image
path = './imgs/baby1.png'
# path = './imgs/group.jpg'
img = mpimg.imread(path)

params_ik = {
    "mapwidth": 64,
    "face_model":'cnn',
}

features = ["intensity", "orientation", "color"]
IK = IttiKoch(input_params=params_ik, verbose=True)
saliency, maps = IK.run(img, features, faces=True)

fig, ax = plt.subplots(2, 3, figsize=(15, 5))
ax[0,0].imshow(img)
ax[0,1].imshow(maps[0])
ax[0,2].imshow(maps[1])
ax[1,0].imshow(maps[2])
ax[1,1].imshow(maps[3])
ax[1,2].imshow(saliency)
ax[0,0].set_title('image')
ax[0,1].set_title('intensity')
ax[0,2].set_title('orientation')
ax[1,0].set_title('color')
ax[1,1].set_title('faces')
ax[1,2].set_title('ICO')

plt.show()
