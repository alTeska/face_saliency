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

warnings.simplefilter(action='ignore', category=FutureWarning)
plt.rcParams['image.cmap'] = 'gray'

path = './data/test/results/'
fnames = glob('./data/test/*.jpg')
print(fnames)

# take care of directories
directories = ['ik']

if not os.path.exists(path):
    os.makedirs(path)

if not os.path.exists('./data/test/done'):
    os.makedirs('./data/test/done')


for direct in directories:
    if not os.path.exists(path+direct):
        print('creating directory ' + direct)
        os.makedirs(path+direct)


# initiate our model
IK = IttiKoch(verbose=False)

for fname in tqdm(fnames):
    name = fname[12:-4]  # get just the name of the picture
    print(name)
    img = mpimg.imread(fname)

    # run basic models
    smap_ik, _ = IK.run(img, faces=False)

    # normalize saliecies
    smap_ik = normalize_saliency_map(smap_ik)


    # save plots
    print('Saving plots')
    print(path + 'ik/'+ name + '.jpg')
    save_plot_without_frames(smap_ik, path + 'ik/' + name + '.jpg')

    shutil.move(fname, 'data/test/done')
