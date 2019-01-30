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

import pysaliency
from pysaliency.utils import MatlabOptions

plt.rcParams['image.cmap'] = 'gray'
MatlabOptions.matlab_names = ['matlab', 'matlab.exe', '/usr/local/MATLAB/R2017b/bin/matlab']
MatlabOptions.octave_names = []


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


path = 'data/face/results/'
fnames = glob('data/face/*.jpg')

# take care of directories
directories = ['faces']

if not os.path.exists(path):
    os.makedirs(path)

for direct in directories:
    if not os.path.exists(path+direct):
        print('creating directory ' + direct)
        os.makedirs(path+direct)


# initiate our model
params_ik = {
    "mapwidth": 64,
    "gaussian_blur": 10,
    "face_model":'cnn',
}

features = ["intensity", "orientation", "color"]
IK = IttiKoch(input_params=params_ik, verbose=False)

for fname in tqdm(fnames):
    name = fname[10:-4]  # get just the name of the picture
    img = mpimg.imread(fname)

    # run basic models
    smap_face, _ = IK.run(img, keys = [], faces=True)

    # normalize saliecies
    smap_face = normalize_saliency_map(smap_face)
    # smap_ik = normalize_saliency_map(smap_ik)

    # save plots
    save_plot_without_frames(smap_face, path + 'faces/' + name + '.jpg')

    shutil.move(fname, 'data/face/done/')

    # pbar.close()
