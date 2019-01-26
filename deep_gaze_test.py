import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import skimage
import matplotlib.image as mpimg
from scipy.misc import face
sns.set_style('white')


# Load The Image
path = './imgs/67.jpg'
img = mpimg.imread(path)
# img = skimage.img_as_float(img)

if np.shape(img)[2] == 4:
    img = img[:,:,0:3]

image_data = img[np.newaxis, :, :, :]  # BHWC, three channels (RGB)
centerbias_data = np.zeros((1, img.shape[0], img.shape[1], 1))  # BHWC, 1 channel (log density)

tf.reset_default_graph()
check_point = './deep_gaze/DeepGazeII.ckpt'  # DeepGaze II
new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))
#check_point = 'ICF.ckpt'  # ICF

# import deepgaze model
input_tensor = tf.get_collection('input_tensor')[0]
centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
log_density = tf.get_collection('log_density')[0]
log_density_wo_centerbias = tf.get_collection('log_density_wo_centerbias')[0]


with tf.Session() as sess:
    new_saver.restore(sess, check_point)

    log_density_prediction = sess.run(log_density, {
        input_tensor: image_data,
        centerbias_tensor: centerbias_data,
    })

print(log_density_prediction.shape)

# visualize the model
plt.figure()
plt.gca().imshow(img, alpha=0.2)
m = plt.gca().matshow((log_density_prediction[0, :, :, 0]), alpha=0.5, cmap=plt.cm.RdBu)
plt.colorbar(m)
plt.title('log density prediction')
plt.axis('off');

plt.figure()
plt.gca().imshow(img, alpha=0.2)
m = plt.gca().matshow(np.exp(log_density_prediction[0, :, :, 0]), alpha=0.5, cmap=plt.cm.RdBu)
plt.colorbar(m)
plt.title('density prediction')
plt.axis('off');

plt.show()
