import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pysaliency
import pysaliency.external_datasets
from pysaliency.utils import MatlabOptions

plt.rcParams['image.cmap'] = 'gray'

MatlabOptions.matlab_names = ['matlab', 'matlab.exe', '/usr/local/MATLAB/R2017b/bin/matlab']
MatlabOptions.octave_names = []

# Load The Image
path = './imgs/balloons.png'
img = mpimg.imread(path)

aim = pysaliency.AIM(location='test_models', cache_location=os.path.join('model_caches', 'AIM'))

smap = aim.saliency_map(img)
plt.imshow(-smap)
# plt.axis('off');

plt.show()
