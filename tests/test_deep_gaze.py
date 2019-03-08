import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
warnings.simplefilter(action='ignore', category=FutureWarning)

from saliency_model.deep_gaze import run_deep_gaze


# Load The Image
path = './imgs/lena.jpg'
img = mpimg.imread(path)

# account for png files
if np.shape(img)[2] == 4:
    img = img[:,:,0:3]

# run the model
smap, log_density_prediction = run_deep_gaze(img)
smap_icf, log_density_prediction_icf = run_deep_gaze(img, model='ICF')

# visualize the model
fig, ax = plt.subplots(1, 2)
# ax[0].imshow(img, alpha=0.2)
# ax[0].imshow(smap, alpha=0.5)
# ax[1].imshow(img, alpha=0.2)
# ax[1].imshow(smap_icf, alpha=0.5)

print(type(smap))
print(type(smap_icf))
# print(smap.dtype())
# print(smap_icf.dtype())

ax[0].imshow(smap)
ax[1].imshow(smap_icf)

plt.show()
