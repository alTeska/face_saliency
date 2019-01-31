import os
import gc
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from tqdm import tqdm

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


path_inp = 'data/face/'
path = 'data/face/results/'
fnames = glob('data/face/*.jpg')

# take care of directories
directories_inp = ['ik', 'aim', 'sun', 'cas', 'covsal', 'gbvs','icf']
directories_out = ['ik_face', 'aim_face', 'sun_face', 'cas_face','covsal_face', 'gbvs_face', 'icf_face']
# directories_inp = ['ik']
# directories_out = ['ik_face']


if not os.path.exists(path_inp+'done/'):
    os.makedirs(path_inp+'done/')

if not os.path.exists(path):
        os.makedirs(path)

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
    name = fname[10:]  # get just the name of the picture
    smap_face = mpimg.imread(fname)

    for direct in tqdm(directories_inp):
        fname_inp = path_inp + direct + '/' + name
        fname_out = path + direct + '_face/' + name

        smap = mpimg.imread(fname_inp)

        # normalize saliecies
        smap = normalize_saliency_map(smap)
        smap_face = normalize_saliency_map(smap_face)

        smap_final_03 = normalize_saliency_map(smap * 0.7 + smap_face * 0.3)
        smap_final_05 = normalize_saliency_map(smap * 0.5 + smap_face * 0.5)
        smap_final_07 = normalize_saliency_map(smap * 0.3 + smap_face * 0.7)

        # fig, ax = plt.subplots(1,5, figsize=(10,10))
        #
        # ax[0].imshow(smap)
        # ax[1].imshow(smap_face)
        # ax[2].imshow(smap_final_03)
        # ax[3].imshow(smap_final_05)
        # ax[4].imshow(smap_final_07)
        # plt.show()

        save_plot_without_frames(smap_final_03, path + 'face_03/' + direct + '_face/' + name)
        save_plot_without_frames(smap_final_05, path + 'face_05/' + direct + '_face/' + name)
        save_plot_without_frames(smap_final_07, path + 'face_07/' + direct + '_face/' + name)

    shutil.move(fname, 'data/face/done/')
