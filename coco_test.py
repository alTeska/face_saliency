from glob import glob
from tqdm import tqdm
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)

import pysaliency
from pysaliency.utils import MatlabOptions
from saliency_model.itti_koch import IttiKoch
from saliency_model.deep_gaze import run_deep_gaze

plt.rcParams['image.cmap'] = 'gray'
MatlabOptions.matlab_names = ['matlab', 'matlab.exe', '/usr/local/MATLAB/R2017b/bin/matlab']
MatlabOptions.octave_names = []


# fnames = glob('./data/test/*.jpg')
fnames = glob('data/test/*.jpg')

# for i, fname in tqdm(enumerate(fnames)):
for fname in tqdm(fnames):
    print(fname)
