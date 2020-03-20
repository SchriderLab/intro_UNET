import matplotlib.pyplot as plt
import numpy as np
import h5py

import os
import logging, argparse

import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/kilgoretrout/intro_UNET/src/data/')

from data_functions import *
import os

import random
from scipy.spatial.distance import pdist, squareform

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "archie_200_data_slim.hdf5")
    parser.add_argument("--format_mode", default = "None")

    parser.add_argument("--odir", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)


    return args

def main():
    args = parse_args()

    ifile = h5py.File(args.ifile, 'r')
    keys = sorted(list(map(int, ifile.keys())))

    random.shuffle(keys)

    for key in keys:
        X = ifile['{0}/x_0'.format(key)]
        Y = ifile['{0}/y'.format(key)]

        for k in range(len(X)):

            x = X[k,:,:,0]
            y = Y[k,:,:,0]

            fig, axes = plt.subplots(nrows=4)
            axes[0].imshow(x[64:,:])
            axes[1].imshow(y[64:,:])

            y, indices = sort_NN(y, method = 'max')
            x = x[indices]

            x = x[64:,:]
            y = y[64:,:]

            axes[2].imshow(x)
            axes[3].imshow(y)

            plt.show()

if __name__ == '__main__':
    main()