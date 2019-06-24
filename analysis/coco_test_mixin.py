import os
import shutil
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from tqdm import tqdm
from saliency_model.itti_koch import IttiKoch
from saliency_model.deep_gaze import run_deep_gaze
from saliency_model.utils import normalize_saliency_map
from utils_analysis import save_plot_without_frames

warnings.simplefilter(action='ignore', category=FutureWarning)

import pysaliency
from pysaliency.utils import MatlabOptions

retval = os.getcwd()
print("Current working directory %s" % retval)


plt.rcParams['image.cmap'] = 'gray'
MatlabOptions.matlab_names = ['matlab', 'matlab.exe', '/usr/local/MATLAB/R2017b/bin/matlab']
MatlabOptions.octave_names = []

path = 'data/redo/results/'
fnames = glob('data/redo/*.jpg')
print(fnames)


# take care of directories
directories = ['IK', 'faces',  'aim', 'sun', 'cas', 'covsal', 'gbvs', 'dg',
               'icf', 'ik_face', 'aim_face', 'sun_face', 'cas_face',
               'covsal_face', 'gbvs_face', 'icf_face']

if not os.path.exists(path):
    os.makedirs(path)

for direct in directories:
    if not os.path.exists(path+direct):
        print('creating directory ' + direct)
        os.makedirs(path+direct)


# initiate our model
IK = IttiKoch(verbose=False)

# initate models from pysaliency
aim = pysaliency.AIM(location='test_models', cache_location=os.path.join('model_caches', 'AIM'))
sun = pysaliency.SUN(location='test_models', cache_location=os.path.join('model_caches', 'SUN'))
cas = pysaliency.ContextAwareSaliency(location='test_models', cache_location=os.path.join('model_caches', 'ContextAwareSaliency'))
covsal = pysaliency.CovSal(location='test_models', cache_location=os.path.join('model_caches', 'CovSal'))
gbvs = pysaliency.GBVS(location='test_models', cache_location=os.path.join('model_caches', 'GBVS'))

fnames = fnames[0:2]
for fname in tqdm(fnames):
    print(fname, '\n')

    name = fname[12:-4]  # get just the name of the picture
    img = mpimg.imread(fname)

    pbar = tqdm(total=100)

    # run basic models
    smap_ik, _ = IK.run(img)
    pbar.update(10)
    smap_face, _ = IK.run(img, keys = [], faces=True)
    pbar.update(10)

    # run pysaliency models
    smap_aim = aim.saliency_map(img)
    pbar.update(10)
    smap_sun = sun.saliency_map(img)
    pbar.update(10)
    smap_cas = cas.saliency_map(img)
    pbar.update(10)
    smap_covsal = covsal.saliency_map(img)
    pbar.update(10)
    smap_gbvs = gbvs.saliency_map(img)
    pbar.update(10)

    # run deep gaze models
    smap_dg, log_density_prediction = run_deep_gaze(img)
    pbar.update(10)
    smap_icf, log_density_prediction_icf = run_deep_gaze(img, model='ICF')
    pbar.update(10)

    # normalize saliecies
    smap_ik = normalize_saliency_map(smap_ik)
    smap_face = normalize_saliency_map(smap_face)
    smap_aim = normalize_saliency_map(smap_aim)
    smap_sun = normalize_saliency_map(smap_sun)
    smap_cas = normalize_saliency_map(smap_cas)
    smap_covsal = normalize_saliency_map(smap_covsal)
    smap_gbvs = normalize_saliency_map(smap_gbvs)
    smap_dg = normalize_saliency_map(smap_dg)
    smap_icf = normalize_saliency_map(smap_icf)

    # add face to other (non-deep learnig) saliencies
    smap_ik_face = normalize_saliency_map(smap_ik + smap_face)
    smap_aim_face = normalize_saliency_map(smap_aim + smap_face)
    smap_sun_face = normalize_saliency_map(smap_sun + smap_face)
    smap_cas_face = normalize_saliency_map(smap_cas + smap_face)
    smap_covsal_face = normalize_saliency_map(smap_covsal + smap_face)
    smap_gbvs_face = normalize_saliency_map(smap_gbvs + smap_face)
    smap_icf_face = normalize_saliency_map(smap_icf + smap_face)

    # save plots
    print('Saving plots')
    save_plot_without_frames(smap_ik, path + 'IK/' + name + '.jpg')
    save_plot_without_frames(smap_face, path + 'faces/' + name + '.jpg')
    save_plot_without_frames(smap_aim, path + 'aim/' + name + '.jpg')
    save_plot_without_frames(smap_sun, path + 'sun/' + name + '.jpg')
    save_plot_without_frames(smap_cas, path + 'cas/' + name + '.jpg')
    save_plot_without_frames(smap_covsal, path + 'covsal/' + name + '.jpg')
    save_plot_without_frames(smap_gbvs, path + 'gbvs/' + name + '.jpg')
    save_plot_without_frames(smap_dg, path + 'dg/' + name + '.jpg')
    save_plot_without_frames(smap_icf, path + 'icf/' + name + '.jpg')

    save_plot_without_frames(smap_ik_face, path + 'ik_face/' + name + '.jpg')
    save_plot_without_frames(smap_aim_face, path + 'aim_face/' + name + '.jpg')
    save_plot_without_frames(smap_sun_face, path + 'sun_face/' + name + '.jpg')
    save_plot_without_frames(smap_cas_face, path + 'cas_face/' + name + '.jpg')
    save_plot_without_frames(smap_covsal_face, path + 'covsal_face/' + name + '.jpg')
    save_plot_without_frames(smap_gbvs_face, path + 'gbvs_face/' + name + '.jpg')
    save_plot_without_frames(smap_icf_face, path + 'icf_face/' + name + '.jpg')

    pbar.update(10)
    # shutil.move(fname, 'data/redo/done')

    pbar.close()
