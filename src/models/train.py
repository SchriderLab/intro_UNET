import sys, os, argparse
import itertools
import configparser
import pickle
import numpy as np
import psutil
import tensorflow as tf

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session
set_session(session)

import sklearn
import keras
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.pooling import MaxPooling2D, AveragePooling1D
from architecture_functions import *

from data_on_the_fly_classes import DataGenerator
from cnn_data_functions import create_data_batch, rm_broken_sims, get_max_snps, get_ids, partition_data, get_partition_indices

from keras.utils import multi_gpu_model
from keras import backend as k
import logging
from keras.callbacks import TensorBoard

from keras.engine.input_layer import InputLayer
import h5py

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def relu_clipped(x):
    return K.relu(x, max_value=1)

print('Python version ' + sys.version)
print('argparse version ' + argparse.__version__)
print('numpy version ' + np.__version__)
print('psutil version ' + psutil.__version__)
print('tensorflow version ' + tf.__version__)
print('sklearn version ' + sklearn.__version__)
print('keras version ' + keras.__version__ + '\n')

def parse_args():
    # Argument Parser    
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--model", default = "UNET_even_smaller.json")
    parser.add_argument("--data", default = "training.data.npz")
    parser.add_argument("--tag", default = "test")
    parser.add_argument("--train_config", default = "training_configs/adam_default", help = "contains the settings for training, the optimizer, learning rate, etc.")

    parser.add_argument("--odir", default = "training_output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.odir):
        os.mkdir(args.odir)
        logging.debug('root: made output directory {0}'.format(args.odir))

    return args

def train_cnn_model(model, configFile, weightFileName, xdata_train, ytarget_train, xdata_test, ytarget_test, gpus = 1):
    """
    Compile the model with the adam optimizer and the mean_squared_error loss function.
    Train the model for a given number of epochs in a given batch size until a specified stopping criterion is reached.
    Retain the best performing CNN as assessed on the validation set.
    :param model:
    :param weightFileName:
    :param trainX:
    :param trainPosX:
    :param trainy:
    :param valX:
    :param valPosX:
    :param valy:
    :return:
    """

    if gpus >= 2:
        model = multi_gpu_model(model, gpus=gpus)

    config = configparser.ConfigParser()
    config.read(configFile)

    # get the optimizer name
    op_name = config.get('optimizer', 'name')

    if op_name == 'adam':
        lr = float(config.get('optimizer_params', 'lr'))
        
        op = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


    n_epochs = int(config.get('optimizer_params', 'n_epochs'))
    n_steps_per_epoch = int(config.get('optimizer_params', 'n_steps_per_epoch'))

    batch_size = int(config.get('optimizer_params', 'batch_size'))
    LROnPlateau_patience = int(config.get('optimizer_params', 'LROnPlateau_patience'))
    LROnPlateau_factor = float(config.get('optimizer_params', 'LROnPlateau_factor'))
    EarlyStoppingPatience = int(config.get('optimizer_params', 'EarlyStoppingPatience'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    rlrop = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = LROnPlateau_factor, patience = LROnPlateau_patience, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience = EarlyStoppingPatience, verbose=0, mode='auto')
    checkpoint = keras.callbacks.ModelCheckpoint(weightFileName, monitor='val_loss', verbose=1, save_best_only=True,
                                                 mode='min')
    callbacks = [checkpoint, rlrop]

    history = model.fit(xdata_train, ytarget_train, batch_size=64,
          epochs=25, verbose=1,
          validation_data=(xdata_test, ytarget_test), callbacks = callbacks)

    return history
    

def main():
    ##### Parse arguments
    args = parse_args()

    # Get some output file names
    weightFileName = os.path.join(str(args.odir), '{0}.weights'.format(str(args.tag)))
    testPredFileName = os.path.join(str(args.odir), '{0}.classes'.format(str(args.tag)))
    modFileName = os.path.join(str(args.odir), '{0}.mdl'.format(str(args.tag)))
    evalFileName = os.path.join(str(args.odir), '{0}.evals'.format(str(args.tag)))
    profileFileName = os.path.join(str(args.odir), '{0}.profile'.format(str(args.tag)))

    ##### Load the data
    data = np.load(args.data)

    xdata = data['x']
    ytarget = data['target']
    s = xdata.shape

    xdata,ytarget = np.reshape(xdata, newshape=(s[0], s[1], s[2], 1)), np.reshape(ytarget, newshape=(s[0], s[1], s[2], 1))

    xdata_train,ytarget_train = xdata[1000:], ytarget[1000:]
    xdata_test, ytarget_test  = xdata[:1000], ytarget[:1000]

    ##### Load the model from the specified JSON file
    model = model_from_json(open(args.model, 'r').read(), custom_objects={'relu_clipped': relu_clipped})

    history = train_cnn_model(model, args.train_config, weightFileName, xdata_train, ytarget_train, xdata_test, ytarget_test)

    model.load_weights(weightFileName)

    p = model.predict(xdata_test)

    for i in range(20):
        plt.subplot(3,1,1)
        plt.imshow(np.reshape(xdata_test[i], (48, 128)), aspect=.5, cmap='bone')

        plt.subplot(3,1,2)
        plt.imshow(np.reshape(ytarget_test[i], (48, 128)), aspect=.5, cmap='bone')

        plt.subplot(3,1,3)
        plt.imshow(np.rint(np.reshape(p[i],  (48, 128))), aspect=.5, cmap='bone')
        plt.savefig('exmaple_{0:03d}.png'.format(i))

        plt.close()

if __name__ == '__main__':
    main()

    

    
