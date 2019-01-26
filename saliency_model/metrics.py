import numpy as np
import os
from skimage.transform import resize
import matplotlib.image as mpimg

from utils import gaussian2D, center_bias


def run_dataset_analysis(gt_paths, sal_paths, skip_auc = False):
    
    # construct the data paths
    #data_path = os.path.join(dataset, "images/train")
    #data_path = os.path.join(path, "pipeline_test")
    #fix_path = os.path.join(path, "fixation_maps/train")
    
    # get the training files
    #filenames = os.listdir(data_path)
    #num_files = len(filenames)
    #print(filenames)
    
    num_files = len(gt_paths)
    
    # initialize the score arrays
    nss = np.full(num_files, np.nan)
    sim = np.full(num_files, np.nan)
    ig = np.full(num_files, np.nan)
    auc = np.full(num_files, np.nan)
    
    sal = sal_paths
    for gt,i in zip(gt_paths, np.arange(num_files)):
    #for gt,sal in zip(gt_paths, sal_paths):
        try:
             sal_map = mpimg.imread(sal)
        except:
            print("Image at path {} could not be found.".format(sal))
            continue

        # get corresponding fixation map
        #filename = f.split('.')[0] + '.png'

        try:
            gt_map = mpimg.imread(gt)
        except:
            print("Fixation Map at path {} could not be found.".format(gt))
            continue

        # compute saliency map with model
        #if sal_model == "Itti Koch": sal_map, temp = itti_koch.run(image)

        nss[i], sim[i], ig[i], auc[i] = compute_all_metrics(sal_map, gt_map, skip_auc = skip_auc)
        print(nss[i], sim[i], ig[i], auc[i])
        
    return np.nanmean(nss), np.nanmean(sim), np.nanmean(ig), np.nanmean(auc)


def compute_all_metrics(sal_map, fix_map = [], fix_binary = [], baseline = [], skip_auc = False):
    '''
    Computes the score-value for all metrics. Only scores computed, for additional information,
    run the single function.
    Output: NSS, SIM, IG, AUC
    '''
    
    fix_binary = fix_binary if list(fix_binary) else fix_map > 0
    fix_map = fix_map if list(fix_map) else fix_binary
        
    # adjust the image sizes for further computations
    sal_map, fix_map = adjust_image_size(sal_map, fix_map)
    
    if skip_auc:
        auc_score = 0
    else:
        auc_score, temp, temp, temp = auc_judd_score(sal_map, fix_binary)
        
    nss = compute_nss(sal_map, fix_binary)
    sim_score = compute_similarity(sal_map, fix_map)
    info_gain = compute_information_gain(sal_map, fix_binary, baseline)
    
    return nss, sim_score, info_gain, auc_score


def auc_judd_score(sal_map, fix_map):
    '''
    Computes the AUC of the saliency map and the fixation map according to Judd et al. 
    TODO: more explanation here?
    '''
    
    # scale both images to the size of fixation map
    sal_map, fix_map = adjust_image_size(sal_map, fix_map)

    # sort the saliency values for the real fixations, which are then used as thresholds
    sorted_th = np.flip(np.sort(sal_map[fix_map > 0]))
    
    # compute the true and false positives
    tp, fp = compute_tp_fp(sorted_th, sal_map)
    
    # compute the area under the curve
    score = np.trapz(tp, fp)
    
    # fill up threshold values for output (corresponding to filling up the tp and fp arrays)
    thresholds = np.append([1], sorted_th, axis=0)
    thresholds = np.append(thresholds, [0], axis = 0)
    
    return score, tp, fp, thresholds


def adjust_image_size(img1, img2, downscale_only = False):
    '''
    Checks if the images have the same size. If not, it rescales the first image to the size of the second image. 
    If downscale_only is set to true, the bigger one is scaled to the size of the smaller image, regardless of the ordering.
    '''
    # make the images the same size
    if downscale_only:
        # scale the bigger map to the smaller size
        if (np.size(img1,0) > np.size(img2,0) or np.size(img1,1) > np.size(img2,1)):
            img1 = resize(img1, np.shape(img2), mode='constant', anti_aliasing=True)
            
        elif (np.size(img1,0) < np.size(img2,0) or np.size(img1,1) < np.size(img2,1)):
            img2 = resize(img2, np.shape(img1), mode='constant', anti_aliasing=True)
    
    else:
        # scale both maps to the size of the first map
        if (np.size(img1, 0) != np.size(img2, 0) or np.size(img1, 1) != np.size(img2, 1)):
            img1 = resize(img1, np.shape(img2), mode='constant', anti_aliasing=True)
    
    return img1, img2 


def compute_tp_fp(sorted_thr, sal_map):
    '''
    Computes the true and false positives corresponding to the sorted thresholds for the ground truth fixations and the given saliency map.
    '''
    
    # amount of pixels in total and amount of true fixations
    n_pixels = np.size(sal_map)
    n_fixations = np.size(sorted_thr)
    
    # arrays for true and false positive rates
    tp = np.zeros(n_fixations+2)
    fp = np.zeros(n_fixations+2)
    
    # these arrays are filled up with a 0 and a 1 at the beginning and end respectively
    # 0 at the beginning: there are no positive values if the threshold is too high
    # 1 at the end: the threshold is zero, so all fixation pixels are true positives but also all not fixated pixels are true negatives

    tp[n_fixations+1] = 1
    fp[n_fixations+1] = 1

    for i in range(0, n_fixations):
        # true positives
        tp[i+1] = i / n_fixations

        # find saliency values above current threshold
        above_thr = np.sum(sal_map > sorted_thr[i])

        # false positives
        fp[i+1] = (above_thr - i) / (n_pixels - n_fixations)
        
    return tp, fp


def compute_nss(sal_map, fix_binary):
    
    # adjust the image size if it hasn't been done before
    sal_map, fix_binary = adjust_image_size(sal_map, fix_binary)
    
    # compute total amount of fixated pixels
    N_fixations = np.sum(fix_binary, axis = (0,1))
    
    # z-score of saliency map
    sal_norm = (sal_map - np.mean(sal_map)) / np.std(sal_map)

    # compute NSS
    NSS = np.sum(np.multiply(sal_norm, fix_binary), axis = (0,1)) / N_fixations
    
    return NSS


def compute_similarity(sal_map, fix_map):
   
    # adjust the image size if it hasn't been done before
    sal_map, fix_map = adjust_image_size(sal_map, fix_map)
    
    # normalize both to match probability distributions
    fix_norm = fix_map / np.sum(fix_map)
    sal_norm = sal_map / np.sum(sal_map)
    
    # sum over the minima of both normalized maps
    return np.sum(np.minimum(fix_norm, sal_norm))


def compute_information_gain(sal_map, fix_binary, baseline = []):
    '''
    Computes the information gain of the saliency map over the baseline. If no baseline is provided,
    the information gain over the center bias is computed.
    '''
    # adjust image size if it hasn't happened before
    sal_map, fix_binary = adjust_image_size(sal_map, fix_binary)
    
    # create center bias baseline if no baseline provided
    baseline = baseline if list(baseline) else center_bias(lambda x, y: gaussian2D(x, y, 100), np.shape(fix_binary))
    
    # bring the maps to probability distribution
    sal_map = sal_map - np.min(sal_map)
    sal_map = sal_map / np.sum(sal_map)
    baseline = baseline - np.min(baseline)
    baseline = baseline / np.sum(baseline)
    
    # compute information gain
    epsilon = np.finfo('float64').eps
    ig = np.sum(fix_binary * (np.log2(epsilon + sal_map) - np.log2(epsilon + baseline))) / np.sum(fix_binary)
    
    return ig


