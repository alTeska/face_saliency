import os
import gc
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from tqdm import tqdm
from saliency_model.utils import normalize_saliency_map
from utils_analysis import save_plot_without_frames


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


plt.rcParams['image.cmap'] = 'gray'

path_inp = './data/redo/'
path = './data/redo/results/'
fnames = glob('./data/redo/*.jpg')

# take care of directories
# directories_inp = ['ik', 'aim', 'sun', 'cas', 'covsal', 'gbvs','icf']
# directories_out = ['ik_face', 'aim_face', 'sun_face', 'cas_face','covsal_face', 'gbvs_face', 'icf_face']
directories_inp = ['ik', 'aim', 'sun', 'covsal', 'gbvs']
directories_out = ['ik', 'aim', 'sun', 'covsal', 'gbvs']


if not os.path.exists(path_inp+'done/'):
    os.makedirs(path_inp+'done/')

if not os.path.exists(path):
        os.makedirs(path)

for direct in directories_out:
    if not os.path.exists(path +'no_face/'+direct):
        print('creating directory ' + direct)
        os.makedirs(path +'no_face/'+direct)

for direct in directories_out:
    if not os.path.exists(path +'face_03/'+direct):
        print('creating directory ' + direct)
        os.makedirs(path +'face_03/'+direct)

for direct in directories_out:
    if not os.path.exists(path+'face_05/'+direct):
        print('creating directory ' + direct)
        os.makedirs(path+'face_05/'+direct)

for direct in directories_out:
    if not os.path.exists(path+'face_07/'+direct):
        print('creating directory ' + direct)
        os.makedirs(path+'face_07/'+direct)


for fname in tqdm(fnames):
    name = fname[12:]  # get just the name of the picture
    smap_face = mpimg.imread(fname)
    if np.size(smap_face.shape)==3:
        smap_face = rgb2gray(smap_face)
    if np.shape(smap_face) is not (369, 492):
        smap_face = cv2.resize(smap_face, (492, 369), interpolation = cv2.INTER_AREA)

    for direct in tqdm(directories_inp):
        fname_inp = path_inp + direct + '/' + name
        fname_out = path + direct + '_face/' + name

        smap = mpimg.imread(fname_inp)

        if np.size(smap.shape)==3:
            smap = rgb2gray(smap)
        if np.shape(smap) is not (369, 492):
            smap = cv2.resize(smap, (492, 369), interpolation = cv2.INTER_AREA)
        # normalize saliecies
        smap = normalize_saliency_map(smap)
        smap_face = normalize_saliency_map(smap_face)

        smap_final_03 = normalize_saliency_map(smap * 0.7 + smap_face * 0.3)
        smap_final_05 = normalize_saliency_map(smap * 0.5 + smap_face * 0.5)
        smap_final_07 = normalize_saliency_map(smap * 0.3 + smap_face * 0.7)
        smap_final = normalize_saliency_map(smap)

        save_plot_without_frames(smap_final, path + 'no_face/' + direct + '/' + name)
        save_plot_without_frames(smap_final_03, path + 'face_03/' + direct + '/' + name)
        save_plot_without_frames(smap_final_05, path + 'face_05/' + direct + '/' + name)
        save_plot_without_frames(smap_final_07, path + 'face_07/' + direct + '/' + name)

    shutil.move(fname, './data/redo/done/')
