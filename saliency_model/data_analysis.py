import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

from itti_koch import IttiKoch
from metrics import auc_judd_score


# Load example Image
path = '../imgs/d91.jpg'
fix_map = mpimg.imread(path)

path = '../imgs/91.jpg'
img = mpimg.imread(path)

IK = IttiKoch({})
sal_map, temp = IK.run(img)


# (1) AUC metric
score, tp, fp, thresholds = auc_judd_score(sal_map, fix_map)

plt.plot(fp, tp)
plt.plot(np.arange(0, 1, 1/len(fp)), np.arange(0,1, 1/len(fp)));