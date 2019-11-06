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
from keras import backend as K
import logging
from keras.callbacks import TensorBoard

from keras.engine.input_layer import InputLayer
import h5py

import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt

def relu_clipped(x):
    return K.relu(x, max_value=1)

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

def train_cnn_model(model, configFile, weightFileName, training_generator, validation_generator, gpus, tf_logdir):
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

    # Tensor-board callback
    tbCallBack = TrainValTensorBoard(log_dir = tf_logdir, histogram_freq = 0, write_graph = True, write_images = True, write_grads = False, update_freq = 'epoch')

    rlrop = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = LROnPlateau_factor, patience = LROnPlateau_patience, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience = EarlyStoppingPatience, verbose=0, mode='auto')
    checkpoint = keras.callbacks.ModelCheckpoint(weightFileName, monitor='val_loss', verbose=1, save_best_only=True,
                                                 mode='min')
    callbacks = [earlystop, rlrop, checkpoint]

    history = model.fit_generator(generator = training_generator, validation_data = validation_generator,
                        epochs = n_epochs,
                        steps_per_epoch = n_steps_per_epoch,
                        verbose = 1,
                        use_multiprocessing = False, workers = 1,
                        callbacks = callbacks)

    return history

def evaluate_cnn_model(model, weightFileName, generator, testPredFileName, modFileName, evalFileName, gpus):
    if gpus >= 2:
        model = multi_gpu_model(model, gpus=gpus)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.load_weights(weightFileName)

    evals = model.evaluate_generator(generator, verbose=1, steps=None)
    print(evals)

    np.savetxt(evalFileName, evals, fmt='%f')

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
    parser.add_argument("--data", default = "data_v1.0.hdf5")
    parser.add_argument("--n_gpus", default = "1")
    parser.add_argument("--gen_size", default = "64")

    parser.add_argument("--val_prop", default = "0.1")
    parser.add_argument("--test_prop", default = "0.1")
    
    parser.add_argument("--tag", default = "test")
    parser.add_argument("--train_config", default = "training_configs/adam_default", help = "contains the settings for training, the optimizer, learning rate, etc.")

    parser.add_argument("--odir", default = "training_output")
    parser.add_argument("--tf_logdir", default = "training_output")

    parser.add_argument("--indices", default = "None")

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

    
def main():
    ##### Parse arguments
    args = parse_args()

    testProp = float(args.test_prop)
    valProp = float(args.val_prop)
    gpus = int(args.n_gpus)
    tf_logdir = args.tf_logdir

    gen_size = int(args.gen_size)

    # Get some output file names
    weightFileName = os.path.join(str(args.odir), '{0}.weights'.format(str(args.tag)))
    testPredFileName = os.path.join(str(args.odir), '{0}.classes'.format(str(args.tag)))
    modFileName = os.path.join(str(args.odir), '{0}.mdl'.format(str(args.tag)))
    evalFileName = os.path.join(str(args.odir), '{0}.evals'.format(str(args.tag)))
    profileFileName = os.path.join(str(args.odir), '{0}.profile'.format(str(args.tag)))
    historyName = os.path.join(str(args.odir), '{0}.history.pkl'.format(str(args.tag)))

    ifiles = [h5py.File(u, 'r') for u in args.data.split(',')]

    ##### Load the model from the specified JSON file
    model = model_from_json(open(args.model, 'r').read())

    if args.indices == "None":
        indices = dict()
        keys = ['train', 'test', 'val']

        for key in keys:
            indices[key] = []

        for ifile in ifiles:
            _ = get_partition_indices(list(ifile.keys()), testProp, valProp)

            for key in _.keys():
                indices[key].append(_[key])
    else:
        indices = pickle.load(open(args.indices, 'rb'))

    n_inputs = 0

    input_shapes = []

    for layer in model.layers:
        if type(layer) == InputLayer:
            n_inputs += 1
            print(layer.input_shape)
            input_shapes.append(layer.input_shape)

    params = {'ifiles': ifiles,
        'n_inputs': n_inputs,
        'gen_size': gen_size,
        'input_shapes': input_shapes,
        }

    print(params)
    print(model.summary())

    # Generators
    training_generator = DataGenerator(indices['train'], **params)
    validation_generator = DataGenerator(indices['val'], **params)
    test_generator = DataGenerator(indices['test'], **params)

    print([u.shape for u in training_generator[0]])

    history = train_cnn_model(model, args.train_config, weightFileName, training_generator, validation_generator, gpus, tf_logdir)
    evaluate_cnn_model(model, weightFileName, test_generator, testPredFileName, modFileName, evalFileName, gpus)

    pickle.dump(history, open(historyName, 'wb'))

if __name__ == '__main__':
    main()

    

    
