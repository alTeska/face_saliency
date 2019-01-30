import os
import numpy as np
import matplotlib.image as mpimg
from pycocotools.coco import COCO
import scipy.io as sio
from tqdm import tqdm

from metrics import compute_all_metrics, center_bias, gaussian2D


def get_dataset_ids(path):
    filenames = os.listdir(path)

    ds_ids = []
    for file in filenames:
        ds_ids.append(int(file.split('_')[2].split('.')[0]))

    return ds_ids


def create_binary_fixation(path):
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

    def __init__(self, dataDir, models, contexts, metrics, coco=None):
        """
        TODO
        :param dataDir:
        :param models:
        :param contexts:
        :param metrics:
        :param coco:
        """
        self.dataDir = dataDir
        self.models = models
        self.contexts = contexts
        self.metrics = metrics
        self.coco = coco if coco else self.instantiate_coco()

        # datapaths for ground truth and saliency predictions
        self.gt_paths = []
        self.sal_paths = []
        self.fix_paths = []

    def instantiate_coco(self):
        """
        TODO
        :return:
        """
        dataType = 'val2014'
        annFile = '{}\\annotations\\instances_{}.json'.format(self.dataDir, dataType)

        # initialize COCO api for instance annotations
        coco = COCO(annFile)
        return coco

    def store_context_saliencymaps(self, context, model):
        """
        TODO
        :param context:
        :param model:
        :return:
        """
        if context == 'none':
            imgIds = self.coco.getImgIds()
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

        # get corresponding paths (in pairs?)
        for img in images:
            filename = img['file_name'].split('.')[0]
            self.gt_paths.append(os.path.join(self.dataDir, "fixation_maps", filename + ".png"))
            self.sal_paths.append(os.path.join(self.dataDir, "predictions", model, img['file_name']))
            self.fix_paths.append(os.path.join(self.dataDir, "fixations", filename + ".mat"))

    def run_context_analysis(self, skip_auc=False):
        """
        Computes the metrics for each combination of contexts and models.
        :param skip_auc: Parameter to skip the computationally exhausting AUC metric
        :return: array of the form (number of contexts) X (number of models) X (number of metrics) with values stored
        in the same order as passed for initialisation
        """
        num_cxt = np.size(self.contexts)
        num_mod = np.size(self.models)
        summary = np.zeros((num_cxt, num_mod, 4))

        for i in np.arange(num_cxt):
            for j in np.arange(num_mod):
                print("Computing metrics for the {} model in the {} context".format(self.models[j], self.contexts[i]))
                self.store_context_saliencymaps(self.contexts[i], self.models[j])
                summary[i][j] = self.run_dataset_analysis(skip_auc)

                with open('filename.txt', mode='wt', encoding='utf-8') as myfile:
                    myfile.write('\n'.join(str(summary[i][j])))

        return summary

    def run_dataset_analysis(self, skip_auc):
        """
        TODO
        :param skip_auc:
        :return:
        """

        num_files = np.size(self.gt_paths)

        # initialize the score arrays
        nss = np.full(num_files, np.nan)
        sim = np.full(num_files, np.nan)
        ig = np.full(num_files, np.nan)
        auc = np.full(num_files, np.nan)
        
        baseline = center_bias(lambda x, y: gaussian2D(x, y, 50), (480, 640))

        for gt, sal, fix, i in tqdm(zip(self.gt_paths, self.sal_paths, self.fix_paths, np.arange(num_files))):
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

            nss[i], sim[i], ig[i], auc[i] = compute_all_metrics(sal_map, metrics=self.metrics, fix_map=gt_map, fix_binary=fix_map, baseline=baseline)

        return np.nanmean(nss), np.nanmean(sim), np.nanmean(ig), np.nanmean(auc)
