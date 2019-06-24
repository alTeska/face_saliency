import os
import gc
import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from tqdm import tqdm
from saliency_model.itti_koch import IttiKoch
from saliency_model.utils import normalize_saliency_map
from utils_analysis import save_plot_without_frames


# from saliency_model.deep_gaze import run_deep_gaze
warnings.simplefilter(action='ignore', category=FutureWarning)

import pysaliency
from pysaliency.utils import MatlabOptions

plt.rcParams['image.cmap'] = 'gray'
MatlabOptions.matlab_names = ['matlab', 'matlab.exe', '/usr/local/MATLAB/R2017b/bin/matlab']
MatlabOptions.octave_names = []


path = 'data/taim/results/'
fnames = glob('data/taim/*.jpg')
print(fnames)

# take care of directories
directories = ['aim', 'aim_face', 'faces', 'done']

if not os.path.exists(path):
    os.makedirs(path)

for direct in directories:
    if not os.path.exists(path+direct):
        print('creating directory ' + direct)
        os.makedirs(path+direct)


# initiate our model
IK = IttiKoch(verbose=False)

# initate models from pysaliency
aim = pysaliency.AIM(location='test_models', caching=False, cache_location=os.path.join('model_caches', 'AIM'))

for fname in tqdm(fnames):
    name = fname[10:-4]  # get just the name of the picture
    img = mpimg.imread(fname)

    pbar = tqdm(total=40)

    # run basic models
    smap_face, _ = IK.run(img, keys = [], faces=True)
    pbar.update(10)

    # run pysaliency models
    smap_aim = aim.saliency_map(img)
    pbar.update(10)

    # normalize saliecies
    smap_face = normalize_saliency_map(smap_face)
    smap_aim = normalize_saliency_map(smap_aim)
    pbar.update(10)

    # add face to other (non-deep learnig) saliencies
    smap_aim_face = normalize_saliency_map(smap_aim + smap_face)

    # save plots
    print('Saving plots')
    save_plot_without_frames(smap_face, path + 'faces/' + name + '.jpg')
    save_plot_without_frames(smap_aim, path + 'aim/' + name + '.jpg')
    save_plot_without_frames(smap_aim_face, path + 'aim_face/' + name + '.jpg')
    pbar.update(10)

    shutil.move(fname, 'data/imag/done')

    pbar.close()
