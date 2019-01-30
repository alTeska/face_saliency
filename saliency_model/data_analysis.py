import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

from context_analysis import get_dataset_ids, ContextAnalysis

# set dataset specific values
dataDir='C:\\Users\\thier\\Documents\\Studium\\MSNE\\18-19 WS\\NISE\\project\\NISE_saliency\\coco'
context = ['animal', 'sports']
models = ['aim_face']

# create context analysis object with the models and contexts to analyse
ca = ContextAnalysis(dataDir, models, context, metrics=[])
summary = ca.run_context_analysis(skip_auc=True)

print(np.shape(summary))

# save to file for further usage
pickle.dump(summary, open("results.p", "wb"))