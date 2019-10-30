"""
This code is based off of https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
import sys
import numpy as np
import psutil
import keras
from keras.utils import to_categorical
import random

#from PIL import Image
import copy

from cnn_data_functions import read_npz_data, transform_data2d, format_label_arrays, calc_num_classes, format_genotype_arrays, n_snps

def profile(prof_option, func, profileFile):
    if(prof_option == True):
        p = psutil.Process()
        with p.oneshot():
            p.cpu_times()  # return cached value
            p.memory_full_info()

        time = p.cpu_times().user
        mem = (float(p.memory_full_info().uss) / 1048576)
        profileFile.write(str(func) + '\t' + str(time) + ' sec\t' + str(mem) + ' Mb\n')
        profileFile.flush()
    return

def create_y(batch_size, n_models):
    ret = np.zeros((batch_size, n_models))

    n_sims_per = batch_size // n_models

    for k in range(n_models):
        ret[k*n_sims_per:(k+1)*n_sims_per,k] = 1.

    return ret

def get_y_vec(index, models):
    ret = np.zeros(len(models))

    ret[index] = 1.

    return ret

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, indices, ifiles, n_inputs, gen_size, input_shapes):
        self.indices = indices
        self.ifiles = ifiles
        self.gen_size = gen_size
        self.n_inputs = n_inputs
        self.input_shapes = input_shapes

        self.o_indices = copy.copy(indices)

        # grab the pre-set batch size
        self.igen_size = ifiles[0]['0/x_0'].shape[0]

        # check for pre-defined y
        if not 'y' in ifiles[0]['0'].keys():
            # y will be the same in every case for categorical, simply a balanced set of answers
            self.y = create_y(self.gen_size, len(ifiles))
        else:
            self.y = None

        if self.igen_size <= self.gen_size:
            self.n_chunks = self.gen_size // len(ifiles) // self.igen_size

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indices = copy.copy(self.o_indices)

        return

    def __len__(self):
        return int(np.floor(min(map(len, self.indices)) / self.n_chunks))

    def __getitem__(self, index):
        X = []

        indices_ = []

        # take some random sample of the chunks in each HDF5 file passed to the generator
        for k in range(len(self.indices)):
            indices_.append(list(np.random.choice(self.indices[k], self.n_chunks, replace = False)))
            self.indices[k] = list(set(self.indices[k]).difference(indices_[-1]))

        for k in range(self.n_inputs):
            t_shape = self.input_shapes[k]
            # set the batch size
            t_shape = (self.gen_size, ) + t_shape[1:]

            x = []

            for i in range(len(self.ifiles)):
                for j in range(self.n_chunks):
                    x.append(np.array(self.ifiles[i]['{0}/x_{1}'.format(indices_[i][j], k)]))
                    
            x = np.vstack(x)
            
            X.append(x)

        # in the case that n_inputs == 1
        if len(X) == 1:
            X = X[0]

        if self.y is None:
            y = []

            for i in range(len(self.ifiles)):
                for j in range(self.n_chunks):
                    y.append(np.array(self.ifiles[i]['{0}/y'.format(indices_[i][j])]))
                    
            y = np.vstack(y)

            y = y.reshape((y.shape[0], y.shape[1], y.shape[2], 1))
        else:
            y = copy.copy(self.y)
                
        return X, y
        

        

        
    
                 
