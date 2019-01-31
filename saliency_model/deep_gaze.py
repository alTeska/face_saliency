import numpy as np
import warnings
import tensorflow as tf
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_deep_gaze(img, model='DeepGaze'):
    ''' function runs DeepGaze or ICF models provided from DeepGazeII library, uses Matthias code '''

    image_data = img[np.newaxis, :, :, :]  # BHWC, three channels (RGB)
    centerbias_data = np.zeros((1, img.shape[0], img.shape[1], 1))  # BHWC, 1 channel (log density)


    if model == 'DeepGaze':
        check_point = '../deep_gaze/DeepGazeII.ckpt'  # DeepGaze II
    elif model == 'ICF':
        check_point = '../deep_gaze/ICF.ckpt'  # ICF
    else:
        return  AssertionError('Wrong name of the model, could not be loaded! Use DeepGaze or ICF')

    tf.reset_default_graph()
    new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))

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

    smap = np.exp(log_density_prediction[0, :, :, 0])  # get dentisty predicion
    smap = smap / np.max(smap)   # normalize

    return smap, log_density_prediction
