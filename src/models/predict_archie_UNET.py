import os
import numpy as np
import logging, argparse

import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score
import itertools
import configparser
import pickle
import numpy as np
import psutil

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session
set_session(session)

import sklearn
import keras
import sys

from keras.activations import relu
from keras.models import Model, model_from_json

from data_on_the_fly_classes import DataGenerator
from cnn_data_functions import *

from keras.utils import multi_gpu_model
from keras import backend as K
import logging
from keras.callbacks import TensorBoard

from keras.engine.input_layer import InputLayer
from keras.models import load_model

import random
import copy

from scipy.spatial.distance import pdist, squareform

positions = pickle.load(open('positions.pkl', 'rb'))
import cv2

def shuffle_indices(X):
    i1 = list(range(X.shape[0] // 2))
    random.shuffle(i1)

    i2 = list(range(X.shape[0] // 2, X.shape[0]))
    random.shuffle(i2)

    return i1 + i2


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--weights", default = "weights/no_sorting_96_mixed_densenet201.singleGPU.weights")
    parser.add_argument("--model", default = "architectures/var_size/densenet201.json")
    parser.add_argument("--data", default = "archie_200_data_test.hdf5")
    parser.add_argument("--indices", default = "None")

    parser.add_argument("--ofile", default = "evaluation_metrics.pkl")

    parser.add_argument("--shuffle", action = "store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    model = model_from_json(open(args.model, 'r').read())
    model.load_weights(args.weights)

    data_file = h5py.File(args.data, 'r')

    if args.indices != 'None':
        indices = pickle.load(open(args.indices, 'rb'))

        keys = map(str, list(indices['test'][0]))
        keys = list(set(list(data_file.keys())).intersection(keys))
    else:
        keys = list(data_file.keys())

    y_true = []
    y_pred = []

    for key in keys:
        print(key)

        X = np.array(data_file[key + '/x_0'])
        Y = np.array(data_file[key + '/y'])



        if args.shuffle:
            for k in range(len(X)):
                x = copy.copy(X[k])
                y = copy.copy(Y[k])

                #fig, axes = plt.subplots(ncols = 2, nrows = 2)
                #axes[0, 0].imshow(x[:,:,0])
                #axes[0, 1].imshow(y[:,:,0])

                indices = shuffle_indices(x)
                x = x[indices]
                y = y[indices]

                #axes[1, 0].imshow(x[:, :, 0])
                #axes[1, 1].imshow(y[:, :, 0])

                #plt.show()

                X[k] = x
                Y[k] = y

        if 'x_windows' in list(data_file[key].keys()):
            Y_pred = np.zeros(Y.shape)
            count = np.zeros(Y.shape)

            X_windows = np.array(data_file[key + '/x_windows'])
            Y_windows = np.array(data_file[key + '/y_windows'])

            Y = Y_windows[:, 127, :, :, :]

            indices = np.array(data_file[key + '/indices'])


            for k in range(len(X_windows)):
                ref = indices[k, 127]

                Y_pred_k = model.predict(X_windows[k])



                for j in range(X_windows.shape[1]):
                    order = [list(indices[k, j]).index(u) for u in ref]
                    Y_pred_k_j = Y_pred_k[j, order, :, :]

                    Y_pred[k, :, list(np.where(positions[j] == 1)[0]), :] += Y_pred_k_j[:, list(np.where(positions[-(j + 1)] == 1)[0]), :].transpose(1, 0, 2)
                    count[k, :, list(np.where(positions[j] == 1)[0]), :] += 1

            Y_pred /= count

        else:
            Y_pred = model.predict(X)



        print(Y.shape, Y_pred.shape)

        y_true.extend(Y.flatten())
        y_pred.extend(Y_pred.flatten())

    print(len(y_true))

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    result = dict()
    result['roc'] = [fpr, tpr]
    result['pr'] = [precision, recall]

    auroc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)

    result['auroc'] = auroc
    result['aupr'] = aupr

    print(auroc, aupr)

    result['confusion matrix'] = confusion_matrix(y_true, np.round(y_pred))
    result['accuracy'] = accuracy_score(y_true, np.round(y_pred))

    pickle.dump(result, open(args.ofile, 'wb'))

if __name__ == '__main__':
    main()
