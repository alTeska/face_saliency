import os
import numpy as np
import matplotlib.image as mpimg
from pycocotools.coco import COCO
import scipy.io as sio
from tqdm import tqdm

from metrics import compute_all_metrics, center_bias, gaussian2D


def get_dataset_ids(path):
    """
    finds all the image IDS, i.e. filenames, that are existent in the given path
    :param path: directory of the image files
    :return: list of image IDs
    """
    filenames = os.listdir(path)

    ds_ids = []
    for file in filenames:
        ds_ids.append(int(file.split('_')[2].split('.')[0]))

    return ds_ids


def create_binary_fixation(path):
    """
    Creates a binary fixation map out of SALICON matfiles, providing the fixation points and the resolution of the image.
    :param path: directory containing matfiles
    :return: binary fixation map
    """
    fix = sio.loadmat(path)
    
    fix_gaze = fix['gaze'][0][0][0]
    x = fix_gaze[:,0].astype(int)   # width
    y = fix_gaze[:,1].astype(int)   # height
    
    height = fix['resolution'][0][0]
    width = fix['resolution'][0][1]
    
    fix_binary = np.zeros((height, width))
    
    for xi,yi in zip(x,y):
        fix_binary[yi-1, xi-1] = 1
    
    return fix_binary


class ContextAnalysis():

    def __init__(self, dataDir, coco=None):
        """
        initializes an instance of the ContextAnalysis class.
        :param dataDir: directory with the images, predictions and ground truth files in respective folders
        :param models: string array with models to score
        :param contexts: string array with contexts to score models on
        :param metrics: string array with scoring metrics to use
        :param coco (optional): instance of a COCO object from the pycocotools toolbox. If empty, a new COCO object will
        be created using the validation set from 2014.
        """
        self.dataDir = dataDir
        self.models = []
        self.contexts = []
        self.metrics = []
        self.baselines = []
        self.coco = coco if coco else self.instantiate_coco()

        # datapaths for ground truth and saliency predictions
        self.gt_paths = []
        self.sal_paths = []
        self.fix_paths = []
        self.base_paths = []

    def instantiate_coco(self):
        """
        Instantiates a COCO object (from the pycocotools toolbox) from the SALICON / COOO validation set from 2014
        :return: COCO object
        """
        dataType = 'val2014'
        annFile = '{}\\annotations\\instances_{}.json'.format(self.dataDir, dataType)

        # initialize COCO api for instance annotations
        coco = COCO(annFile)
        return coco

    def run_context_analysis(self, models, contexts, metrics, baseline_models, return_all=False, numPictures=None):
        """
        Computes the metrics for each combination of contexts and models.
        :param models:
        :param contexts:
        :param metrics:
        :param baseline_models: array of strings containing corresponding baseline models
        :return:
        """

        num_cxt = np.size(contexts)
        num_mod = np.size(models)

        if return_all:
            summary = np.zeros((num_mod, 4, numPictures))
        else:
            summary = np.zeros((num_cxt, num_mod, 4))

        assert np.size(baseline_models) == np.size(models)

        for i in np.arange(num_cxt):
            for j in np.arange(num_mod):
                self.store_context_saliencymaps(contexts[i], models[j], baseline_models[j])
                if return_all:
                    summary[j] = self.run_dataset_analysis(metrics, baseline_models[j], return_all)
                else:
                    summary[i][j] = self.run_dataset_analysis(metrics, baseline_models[j], return_all)

        return summary, ['nss', 'sim', 'ig', 'auc']

    def store_context_saliencymaps(self, context, model, baseline):
        """
        Filters the given paths from the object directory by context and stores them in the ContextAnalysis object.
        :param context: String specifying COCO category
        :param model: String specifying model folder to look into
        """
        if context == 'all':
            imgIds = self.coco.getImgIds()

        elif context == 'no person':

            # all image Ids
            all_img_ids = set(self.coco.getImgIds())

            # get all categories associated with persons
            cat_ids = self.coco.getCatIds(catNms=context)
            person_img_ids = set(self.coco.getImgIds(catIds=cat_ids))

            # take the difference of both datasets
            imgIds = list(all_img_ids.difference(person_img_ids))

        else:
            # get all categories associated with context
            ids = self.coco.getCatIds(supNms=context)

            # loop through the categories and get corresponding image IDs
            imgIds = []
            for curr_id in ids:
                imgIds = np.append(imgIds, self.coco.getImgIds(catIds=curr_id), axis=0)

        # delete duplicates in case there are some
        coco_ids = list(set(imgIds))

        # just take the pictures that we actually have results for!
        ds_ids = get_dataset_ids(os.path.join(self.dataDir, "predictions", model))
        final_ids = np.intersect1d(coco_ids, ds_ids).astype(int)

        # get images corresponding to ids
        images = self.coco.loadImgs(final_ids)
        print("Context: {}, Model: {}, Baseline: {}, Images found: {}".format(context, model, baseline, len(images)))
        
        self.gt_paths = []
        self.sal_paths = []
        self.fix_paths = []
        self.base_paths = []

        # get corresponding paths
        for img in images:
            filename = img['file_name'].split('.')[0]
            self.gt_paths.append(os.path.join(self.dataDir, "fixation_maps", filename + ".png"))
            self.sal_paths.append(os.path.join(self.dataDir, "predictions", model, img['file_name']))
            self.fix_paths.append(os.path.join(self.dataDir, "fixations", filename + ".mat"))

            if baseline != 'center':
                self.base_paths.append(os.path.join(self.dataDir, "predictions", baseline, img['file_name']))

    def run_dataset_analysis(self, metrics, baseline, return_all):
        """
        Takes the context and model specific image paths and computes all specified metrics on them.
        :return: mean of the metrics over all images, order: NSS, SIM, IG, AUC
        """

        num_files = np.size(self.gt_paths)

        # initialize the score arrays
        nss = np.full(num_files, np.nan)
        sim = np.full(num_files, np.nan)
        ig = np.full(num_files, np.nan)
        auc = np.full(num_files, np.nan)

        for gt, sal, fix, i in tqdm(zip(self.gt_paths, self.sal_paths, self.fix_paths, np.arange(num_files))):

            if baseline == 'none':
                base_model = []
            elif baseline == 'center':
                base_model = center_bias(lambda x, y: gaussian2D(x, y, 50), (480, 640))
            else:
                try:
                    base_model = mpimg.imread(self.base_paths[i])
                except FileNotFoundError:
                    print("Baseline image at path {} could not be found.".format(sal))
                    continue
                except OSError:
                    print("File could not be read. Please make sure to indicate an image file.")
                    continue

            try:
                sal_map = mpimg.imread(sal)
            except FileNotFoundError:
                print("Image at path {} could not be found.".format(sal))
                continue
            except OSError:
                print("File could not be read. Please make sure to indicate an image file.")
                continue

            try:
                gt_map = mpimg.imread(gt)
            except FileNotFoundError:
                print("Fixation Map at path {} could not be found.".format(gt))
                continue
            except OSError:
                print("File could not be read. Please make sure to indicate an image file.")
                continue
                
            try:
                fix_map = create_binary_fixation(fix)
            except FileNotFoundError:
                print("Fixation Matfile at path {} could not be found.".format(fix))
                continue

            # convert rgb to grayscale if necessary
            if np.size(np.shape(sal_map)) == 3:
                sal_map = np.mean(sal_map, axis=2)
            if np.size(np.shape(gt_map)) == 3:
                gt_map = np.mean(gt_map, axis=2)
            if np.size(np.shape(base_model)) == 3:
                base_model = np.mean(base_model, axis=2)

            if not baseline == 'none':
                try:
                    assert(np.size(np.shape(sal_map)) == 2 and np.size(np.shape(gt_map)) == 2
                            and np.size(np.shape(base_model)) == 2)
                except AssertionError:
                    print("Saliency map at path {} has shape {}, groundtruth map at path {} has shape {}, baseline at path"
                          "{} has shape {}."
                          .format(sal, np.shape(sal_map), fix, np.shape(fix_map), self.base_paths[i], np.shape(base_model)))
                    continue

            nss[i], sim[i], ig[i], auc[i] = compute_all_metrics(sal_map,
                                                                metrics=metrics,
                                                                fix_map=gt_map,
                                                                fix_binary=fix_map,
                                                                baseline=base_model)
        if return_all:
            return nss, sim, ig, auc
        else:
            return np.nanmean(nss), np.nanmean(sim), np.nanmean(ig), np.nanmean(auc)
