import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import skimage
import pysaliency
import pysaliency.external_datasets
from pysaliency.utils import MatlabOptions
from saliency_model.itti_koch import IttiKoch


plt.rcParams['image.cmap'] = 'gray'

MatlabOptions.matlab_names = ['matlab', 'matlab.exe', '/usr/local/MATLAB/R2017b/bin/matlab']
MatlabOptions.octave_names = []

# Load The Image
path = './imgs/balloons.png'
img = mpimg.imread(path)
img = skimage.img_as_float(img)

# initiate our model
IK = IttiKoch(verbose=False)
saliency, _ = IK.run(img)


# initate models
aim = pysaliency.AIM(location='test_models', cache_location=os.path.join('model_caches', 'AIM'))
sun = pysaliency.SUN(location='test_models', cache_location=os.path.join('model_caches', 'SUN'))
cas = pysaliency.ContextAwareSaliency(location='test_models', cache_location=os.path.join('model_caches', 'ContextAwareSaliency'))
covsal = pysaliency.CovSal(location='test_models', cache_location=os.path.join('model_caches', 'CovSal'))
gbvs = pysaliency.GBVS(location='test_models', cache_location=os.path.join('model_caches', 'GBVS'))
gbvs_ik = pysaliency.GBVSIttiKoch(location='test_models', cache_location=os.path.join('model_caches', 'GBVS'))

# bms = pysaliency.BMS(location='test_models', cache_location=os.path.join('model_caches', 'BMS'))

# judd = pysaliency.Judd(location='test_models', cache_location=os.path.join('model_caches', 'Judd'))
# rare2012 = pysaliency.RARE2012(location='test_models', cache_location=os.path.join('model_caches', 'RARE2012'))
# IK = pysaliency.IttiKoch(location='test_models', cache_location=os.path.join('model_caches', 'IttiKoch'))


# calculate saliency
smap_aim = aim.saliency_map(img)
smap_sun = sun.saliency_map(img)
smap_cas = cas.saliency_map(img)
smap_covsal = covsal.saliency_map(img)
smap_gbvs = gbvs.saliency_map(img)
smap_gbvs_ik = gbvs_ik.saliency_map(img)

# smap_bms = bms.saliency_map(img)            # Cpp problem (test on windows)

# smap_ik = IK.saliency_map(img)              # requires SaliencyToolbox
# smap_judd = judd.saliency_map(img)          # requires SaliencyToolbox   + TODO: patches need to be applied
# smap_rare2012 = rare2012.saliency_map(img)  # requires SaliencyToolbox   + TODO: problem with link


fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
ax[0][0].imshow(img)
ax[0][1].imshow(saliency)
ax[0][2].imshow(-smap_aim)
ax[0][3].imshow(-smap_sun)
ax[1][0].imshow(smap_cas)
ax[1][1].imshow(smap_gbvs)
ax[1][2].imshow(smap_gbvs_ik)
ax[1][3].imshow(smap_covsal)

ax[0][1].set_title('IttiKoch_ours')
ax[0][2].set_title('AIM')
ax[0][3].set_title('SUN')
ax[1][0].set_title('CAS')
ax[1][1].set_title('GBVS')
ax[1][2].set_title('GBVS_IK')
ax[1][3].set_title('CovSal')



plt.show()
