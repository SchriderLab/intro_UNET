import sys, os, argparse
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

from keras.activations import relu

def relu6(x):
    return relu(x, max_value=6)

from keras.models import Model, model_from_json

from data_on_the_fly_classes import DataGenerator
from cnn_data_functions import *

from keras.utils import multi_gpu_model
from keras import backend as K
import logging
from keras.callbacks import TensorBoard

from keras.engine.input_layer import InputLayer
from keras.models import load_model
import h5py

def relu_clipped(x):
    return K.relu(x, max_value=1)

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

    if config.get('optimizer_params', 'n_steps_per_epoch') != 'None':
        n_steps_per_epoch = int(config.get('optimizer_params', 'n_steps_per_epoch'))
    else:
        n_steps_per_epoch = None

    batch_size = int(config.get('optimizer_params', 'batch_size'))
    LROnPlateau_patience = int(config.get('optimizer_params', 'LROnPlateau_patience'))
    LROnPlateau_factor = float(config.get('optimizer_params', 'LROnPlateau_factor'))
    EarlyStoppingPatience = int(config.get('optimizer_params', 'EarlyStoppingPatience'))

    if config.get('optimizer_params', 'loss') == 'binary_crossentropy':
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', dice_coef_loss, 'binary_crossentropy', mixed_loss])
    elif config.get('optimizer_params', 'loss') == 'dice_coef':
        model.compile(loss=dice_coef_loss,
                      optimizer='adam',
                      metrics=['accuracy', dice_coef_loss, 'binary_crossentropy', mixed_loss])
    elif config.get('optimizer_params', 'loss') == 'mixed':
        model.compile(loss=mixed_loss,
                      optimizer='adam',
                      metrics=['accuracy', dice_coef_loss, 'binary_crossentropy', mixed_loss])
    elif config.get('optimizer_params', 'loss') == 'categorical_crossentropy':
        model.compile(loss='categorical_crossentropy',
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
                  metrics=['accuracy', dice_coef])

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
    parser.add_argument("--weights", default = "None")

    parser.add_argument("--trim", default = "0")

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
    singleGPUweightFileName = os.path.join(str(args.odir), '{0}.singleGPU.weights'.format(str(args.tag)))
    testPredFileName = os.path.join(str(args.odir), '{0}.classes'.format(str(args.tag)))
    modFileName = os.path.join(str(args.odir), '{0}.mdl'.format(str(args.tag)))
    evalFileName = os.path.join(str(args.odir), '{0}.evals'.format(str(args.tag)))
    profileFileName = os.path.join(str(args.odir), '{0}.profile'.format(str(args.tag)))
    historyName = os.path.join(str(args.odir), '{0}.history.pkl'.format(str(args.tag)))

    ifiles = [h5py.File(u, 'r') for u in args.data.split(',')]

    ##### Load the model from the specified JSON file

    if args.weights == 'None':
        model = model_from_json(open(args.model, 'r').read())
    else:
        model = model_from_json(open(args.model, 'r').read())
        model.load_weights(args.weights)

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
        'trim': int(args.trim)
        }

    print(params)
    print(model.summary())

    # Generators
    training_generator = DataGenerator(indices['train'], **params)
    validation_generator = DataGenerator(indices['val'], **params)
    test_generator = DataGenerator(indices['test'], **params)

    print(len(training_generator), len(validation_generator), len(test_generator))

    history = train_cnn_model(model, args.train_config, weightFileName, training_generator, validation_generator, gpus, tf_logdir)
    evaluate_cnn_model(model, weightFileName, test_generator, testPredFileName, modFileName, evalFileName, gpus)

    pickle.dump(history.history, open(historyName, 'wb'))

    # if we have multiple GPUs, save the single-GPU weights
    if gpus > 1:
        multi_gpus_model = load_model(weightFileName, custom_objects={'mixed_loss': mixed_loss, 'dice_coef_loss': dice_coef_loss})
        origin_model = multi_gpus_model.layers[-2]  # you can use multi_gpus_model.summary() to see the layer of the original model
        origin_model.save_weights(singleGPUweightFileName)

if __name__ == '__main__':
    main()

#data/data_AB_NN.hdf5,data/data_BA_NN.hdf5,data/data_bi_NN.hdf5

    
