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
# from saliency_model.deep_gaze import run_deep_gaze
warnings.simplefilter(action='ignore', category=FutureWarning)

# import pysaliency
# from pysaliency.utils import MatlabOptions

plt.rcParams['image.cmap'] = 'gray'
# MatlabOptions.matlab_names = ['matlab', 'matlab.exe', '/usr/local/MATLAB/R2017b/bin/matlab']
# MatlabOptions.octave_names = []


def save_plot_without_frames(img, directory):
    fig, ax = plt.subplots(1,1)

    ax.imshow(img)
    ax.set_axis_off()
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig.savefig(directory, bbox_inches='tight', pad_inches=0, transparent=True);

    plt.close(fig)
    gc.collect()
    pass


def normalize_saliency_map(smap):
    if not (np.max(smap) == 0):  # don't normalize if saliency map is empty
        smap = smap / np.max(smap)
    return smap


path = 'data/redo/results/'
fnames = glob('data/redo/*.jpg')

# take care of directories
directories = ['ik']

if not os.path.exists(path):
    os.makedirs(path)

if not os.path.exists('data/redo/done'):
    os.makedirs('data/redo/done')


for direct in directories:
    if not os.path.exists(path+direct):
        print('creating directory ' + direct)
        os.makedirs(path+direct)


# initiate our model
IK = IttiKoch(verbose=False)


for fname in tqdm(fnames):
    name = fname[10:-4]  # get just the name of the picture
    img = mpimg.imread(fname)

    # run basic models
    smap_ik, _ = IK.run(img, faces=False)

    # run pysaliency models

    # normalize saliecies
    smap_ik = normalize_saliency_map(smap_ik)


    # save plots
    print('Saving plots')
    save_plot_without_frames(smap_ik, path + 'ik/' + name + '.jpg')

    shutil.move(fname, 'data/redo/done')
