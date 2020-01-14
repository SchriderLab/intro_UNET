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

    ifile = h5py.File(args.ifile, 'r+')
    keys = list(ifile.keys())

    for key in keys:
        X_batch = np.array(ifile[key + '/x_0'])
        y_batch = np.array(ifile[key + '/y'])
        feature_batch = np.array(ifile[key + '/features'])

        logging.debug('0: working on key {0}'.format(key))

        for k in range(len(X_batch)):
            X = copy.copy(X_batch[k,:,:,0])
            y = copy.copy(y_batch[k,:,:,0])
            f = copy.copy(f[k,:,:,:])

            if args.format_mode == 'sort_NN':
                X, indices = sort_NN(X)

            elif args.format_mode == 'min_match_sorted':
                X, indices = sort_cdist(x[k], opt='min', sort_pop=True)

            elif args.format_mode == 'max_match_sorted':
                X, indices = sort_cdist(x[k], opt='max', sort_pop=True)

            elif args.format_mode == 'min_match':
                X, indices = sort_cdist(x[k], opt='min', sort_pop=False)

            elif args.format_mode == 'max_match':
                X, indices = sort_cdist(x[k], opt='max', sort_pop=False)

            f = f[:,indices[len(indices) // 2:],:]
            y = y[indices]

            X_batch[k,:,:,0] = X
            y_batch[k,:,:,0] = y
            feature_batch[k,:,:,:] = f

        data_X = ifile[key + '/x_0']
        data_X[...] = X_batch

        data_y = ifile[key + '/y']
        data_y[...] = y_batch

        data_features = ifile[key + '/features']
        data_features[...] = feature_batch

    ifile.close()

if __name__ == '__main__':
    main()



