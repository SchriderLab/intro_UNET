import os
import numpy as np
import logging, argparse

from data_functions import *

import h5py
import copy

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--format_mode", default = "sort_NN")

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
    ofile = h5py.File(args.ofile, 'w')

    keys = list(ifile.keys())

    for key in keys:
        X_batch = np.array(ifile[key + '/x_0'])
        y_batch = np.array(ifile[key + '/y'])

        for k in range(len(X_batch)):
            X = copy.copy(X_batch[k,:,:,0])
            y = copy.copy(y_batch[k,:,:,0])

            if args.format_mode == 'sort_NN':
                X, indices = sort_NN(X)

            elif args.format_mode == 'sort_NN_max':
                X, indices = sort_NN(X, method = 'max')

            elif args.format_mode == 'min_match_sorted':
                X, indices = sort_cdist(X, opt='min', sort_pop=True)

            elif args.format_mode == 'max_match_sorted':
                X, indices = sort_cdist(X, opt='max', sort_pop=True)

            elif args.format_mode == 'min_match':
                X, indices = sort_cdist(X, opt='min', sort_pop=False)

            elif args.format_mode == 'max_match':
                X, indices = sort_cdist(X, opt='max', sort_pop=False)

            y = y[indices]

            X_batch[k,:,:,0] = X
            y_batch[k,:,:,0] = y

        ofile.create_dataset(key + '/x_0', data = X_batch, compression = 'lzf')
        ofile.create_dataset(key + '/y', data = y_batch, compression = 'lzf')

    ifile.close()
    ofile.close()

if __name__ == '__main__':
    main()