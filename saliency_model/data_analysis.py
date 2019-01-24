import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

from itti_koch import IttiKoch
from metrics import auc_judd_score


# Load example Image
path = '../imgs/d91.jpg'
fix_map = mpimg.imread(path)

# create prediction
path = '../imgs/91.jpg'
img = mpimg.imread(path)

IK = IttiKoch({})
sal_map, temp = IK.run(img)

# adjust the image sizes for further computations
sal_map, fix_map = adjust_image_size(sal_map, fix_map)
# create binary fixation
fix_binary = fix_map > 0

# (1) AUC metric
auc_score, tp, fp, thresholds = auc_judd_score(sal_map, fix_binary)
print("AUC: {}".format(auc_score))

plt.plot(fp, tp)
plt.plot(np.arange(0, 1, 1/len(fp)), np.arange(0,1, 1/len(fp)));

# (2) NSS metric
nss_score = compute_nss(sal_map, fix_binary)
print("NSS: {}".format(nss_score))

# (3) Similarity metric
sim_score = compute_similarity(sal_map, fix_map)
print("SIM: {}".format(sim_score))

# (4) Information Gain:
ig = compute_information_gain(sal_map, fix_binary)
print("Information Gain: {}".format(ig))

# compute all of them:
# nss, sim, info_gain, auc = compute_all_metrics(sal_map=sal_map, fix_map=fix_map)