import sys, os, argparse, logging
import numpy as np
import itertools
import keras
from keras import backend as k
import tensorflow as tf
from keras.engine.input_layer import InputLayer
import h5py

import pandas as pd
import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score

from keras.models import load_model

from sklearn.metrics import log_loss
from keras.models import Model, model_from_json

from cnn_data_functions import *
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
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--weights", default = "weights/densenet169_48_final.weights")
    parser.add_argument("--model", default = "architectures/var_size/densenet169.json")

    parser.add_argument("--indices", default = "indices/128/128_10e5_all.pkl")

    parser.add_argument("--data", default = "data/sort_NN/AB.hdf5,data/sort_NN/BA.hdf5,data/sort_NN/bi.hdf5")
    parser.add_argument("--ix", default = "None")

    parser.add_argument("--ofile", default = "AB_metrics.pkl")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    ifiles = [h5py.File(u, 'r') for u in args.data.split(',')]

    indices = pickle.load(open(args.indices, 'rb'))

    model = model_from_json(open(args.model, 'r').read())
    model.load_weights(args.weights)

    model.compile(loss = mixed_loss,
                  optimizer='adam',
                  metrics=['accuracy'])

    n_inputs = 0
    input_shapes = []

    for layer in model.layers:
        if type(layer) == InputLayer:
            n_inputs += 1
            print(layer.input_shape)
            input_shapes.append(layer.input_shape)

    if args.ix == "None":
        params = {'ifiles': ifiles,
                  'n_inputs': n_inputs,
                  'gen_size': 24,
                  'input_shapes': input_shapes,
                  }

        generator = DataGenerator(indices['test'], get_params = True, **params)
    else:
        params = {'ifiles': ifiles,
                  'n_inputs': n_inputs,
                  'gen_size': 8,
                  'input_shapes': input_shapes,
                  }

        generator = DataGenerator([indices['test'][int(args.ix)]], get_params=True, **params)

    y_true = []
    y_pred = []

    for ix in range(50):
        X, y, params = generator[ix]

        y_ = model.predict(X)

        y_true.extend(y.flatten())
        y_pred.extend(y_.flatten())

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    result = dict()
    result['roc'] = [fpr, tpr]
    result['pr'] = [precision, recall]

    auroc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)

    result['auroc'] = auroc
    result['aupr'] = aupr

    result['confusion matrix'] = confusion_matrix(y_true, np.round(y_pred))
    result['accuracy'] = accuracy_score(y_true, np.round(y_pred))

    plt.rc('font', family='Arial', size=11)  # set to Arial/Helvetica
    plt.rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(8, 8), dpi=100)

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fpr, tpr)
    ax.plot([0,1], [0, 1])

    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')

    plt.show()
    plt.close()

    plt.rc('font', family='Arial', size=11)  # set to Arial/Helvetica
    plt.rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(8, 8), dpi=100)

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(recall, precision)

    ax.set_xlabel('recall')
    ax.set_ylabel('precision')

    plt.show()
    plt.close()

    pickle.dump(result, open(args.ofile, 'wb'))

if __name__ == '__main__':
    main()


