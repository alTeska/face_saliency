import os
import shutil
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from tqdm import tqdm
from saliency_model.itti_koch import IttiKoch
from saliency_model.utils import normalize_saliency_map
from utils_analysis import save_plot_without_frames
warnings.simplefilter(action='ignore', category=FutureWarning)

import pysaliency
from pysaliency.utils import MatlabOptions

plt.rcParams['image.cmap'] = 'gray'
MatlabOptions.matlab_names = ['matlab', 'matlab.exe', '/usr/local/MATLAB/R2017b/bin/matlab']
MatlabOptions.octave_names = []

path = '../data/test2/results/'
fnames = glob('../data/test2/*.jpg')

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
