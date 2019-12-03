import sys, os, argparse, logging
import numpy as np
import itertools
import keras
from keras import backend as k
import tensorflow as tf
import h5py

import pandas as pd

from sklearn.metrics import log_loss
from keras.models import Model, model_from_json

from cnn_data_functions import create_data_batch, rm_broken_sims, get_max_snps, get_ids, partition_data, dice_coef
from data_on_the_fly_classes import *

print('Python version ' + sys.version)
print('argparse version ' + argparse.__version__)
print('numpy version ' + np.__version__)
# print('psutil version ' + psutil.__version__)
print('tensorflow version ' + tf.__version__)
# print('sklearn version ' + sklearn.__version__)
print('keras version ' + keras.__version__ + '\n')

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
# config.gpu_options.per_process_gpu_memory_fraction = 0.5

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def binary_crossentropy(y, p):
    return np.mean(-(y*np.log(p) + (1 - y)*np.log(1 - p)))

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action = "store_true")
    parser.add_argument("--weights", default = "None")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--model_file", default = "deepintraSV_k0_64.json")
    parser.add_argument("--ofile", default = "data_AB_evals.csv")

    args = parser.parse_args()

    if args.verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    ifile = h5py.File(args.ifile, 'r')
    keys = sorted(ifile.keys())

    model = model_from_json(open(args.model_file, 'r').read())
    model.load_weights(args.weights)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', dice_coef])

    result = dict()
    result['mig_p1'] = list()
    result['mig_p2'] = list()
    result['mig_time'] = list()
    result['accuracy'] = list()
    result['binary_crossentropy'] = list()
    result['dice_similarity'] = list()

    for key in keys:
        X = np.array(ifile[key]['x_0'], dtype = np.float32)
        y = np.array(ifile[key]['y'], dtype = np.float32)

        params = np.array(ifile[key]['params'])

        for k in range(len(X)):
            X_ = X[k].reshape((1, ) + X[k].shape)
            y_ = y[k].reshape((1, ) + y[k].shape)

            mt, m1, m2 = params[k]

            bc, acc, ds = model.evaluate(X_, y_)

            print(bc)

            result['mig_p1'].append(m1)
            result['mig_p2'].append(m2)
            result['mig_time'].append(mt)
            result['accuracy'].append(acc)
            result['binary_crossentropy'].append(bc)
            result['dice_similarity'].append(ds)

    df = pd.DataFrame(result)
    df.to_csv(args.ofile, header = True, index = False)

if __name__ == '__main__':
    main()