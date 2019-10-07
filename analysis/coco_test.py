import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from tqdm import tqdm
from saliency_model.itti_koch import IttiKoch
from saliency_model.deep_gaze import run_deep_gaze
from utils_analysis import save_plot_without_frames

warnings.simplefilter(action='ignore', category=FutureWarning)

import pysaliency
from pysaliency.utils import MatlabOptions


plt.rcParams['image.cmap'] = 'gray'
MatlabOptions.matlab_names = ['matlab', 'matlab.exe', '/usr/local/MATLAB/R2017b/bin/matlab']
MatlabOptions.octave_names = []

path = 'data/test3/results/'
fnames = glob('data/test3/*.jpg')

# take care of directories
directories = ['dg', 'icf']

if not os.path.exists(path):
    os.makedirs(path)

for direct in directories:
    if not os.path.exists(path+direct):
        print('creating directory ' + direct)
        os.makedirs(path+direct)


# initiate our model
IK = IttiKoch(verbose=False)

location = 'test_models'
location_cache = 'model_caches'

# initate models from pysaliency
# aim = pysaliency.AIM(location=location, caching=False, cache_location=os.path.join(location_cache, 'AIM'))
# sun = pysaliency.SUN(location=location, caching=False, cache_location=os.path.join(location_cache, 'SUN'))
# cas = pysaliency.ContextAwareSaliency(location=location, caching=False, cache_location=os.path.join(location_cache, 'ContextAwareSaliency'))
# covsal = pysaliency.CovSal(location=location, caching=False, cache_location=os.path.join(location_cache, 'CovSal'))
# gbvs = pysaliency.GBVS(location=location, caching=False, cache_location=os.path.join(location_cache, 'GBVS'))

for fname in tqdm(fnames):
    print(fname)
    name = fname[10:-4]  # get just the name of the picture
    img = mpimg.imread(fname)

    # pbar = tqdm(total=90)

    # run basic models
    # smap_ik, _ = IK.run(img)
    # pbar.update(10)
    # smap_face, _ = IK.run(img, keys = [], faces=True)
    # pbar.update(10)

    # run pysaliency models
    # smap_aim = aim.saliency_map(img)
    # pbar.update(10)
    # smap_sun = sun.saliency_map(img)
    # pbar.update(10)
    # smap_cas = cas.saliency_map(img)
    # pbar.update(10)
    # smap_covsal = covsal.saliency_map(img)
    # pbar.update(10)https://marketplace.web.cern.ch/t/womens-salomon-ski-boots-40-chf/32717
    # smap_gbvs = gbvs.saliency_map(img)
    # pbar.update(10)

    # run deep gaze models
    smap_dg, log_density_prediction = run_deep_gaze(img)
    # pbar.update(10)
    smap_icf, log_density_prediction_icf = run_deep_gaze(img, model='ICF')
    # pbar.update(10)
    #
    #
    # save_plot_without_frames(smap_ik, path + 'IK/' + name + '.jpg')
    # save_plot_without_frames(smap_face, path + 'faces/' + name + '.jpg')
    # save_plot_without_frames(smap_aim, path + 'aim/' + name + '.jpg')
    # save_plot_without_frames(smap_sun, path + 'sun/' + name + '.jpg')
    # save_plot_without_frames(smap_cas, path + 'cas/' + name + '.jpg')
    # save_plot_without_frames(smap_covsal, path + 'covsal/' + name + '.jpg')
    # save_plot_without_frames(smap_gbvs, path + 'gbvs/' + name + '.jpg')
    save_plot_without_frames(smap_dg, path + 'dg/' + name + '.jpg')
    save_plot_without_frames(smap_icf, path + 'icf/' + name + '.jpg')
    shutil.move(fname, './data/redo/done/')
#
    # pbar.close()
