# test gabor filters
from saliency_model.utils import create_gabor_kernels
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'


kernels = create_gabor_kernels()

fig, ax = plt.subplots(1, 4, figsize=(15, 15))
ax[0].imshow(kernels[0])
ax[1].imshow(kernels[1])
ax[2].imshow(kernels[2]);
ax[3].imshow(kernels[3]);

plt.show()
