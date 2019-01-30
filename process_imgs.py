import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from saliency_model.itti_koch import IttiKoch

plt.rcParams['image.cmap'] = 'gray'

# Load The Image
# path = './imgs/baby1.png'
path = './data/coco/images/COCO_val2014_000000000675.jpg'

img = mpimg.imread(path)
print(np.shape(img))

params_ik = {
    "mapwidth": 64,
    "gaussian_blur": 20,
}

features = ["intensity", "orientation", "color"]
IK = IttiKoch(input_params=params_ik)
saliency, maps = IK.run(img, features, faces=True)

print(np.shape(saliency))

fig, ax = plt.subplots(1, 4, figsize=(15, 5))
ax[0].imshow(maps[0])
ax[1].imshow(maps[1])
ax[2].imshow(maps[2])
ax[3].imshow(saliency)
ax[0].set_title('intensity')
ax[1].set_title('orientation')
ax[2].set_title('color')
ax[3].set_title('ICO')



plt.show()
