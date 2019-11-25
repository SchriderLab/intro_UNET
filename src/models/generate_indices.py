import os
import numpy as np
import logging, argparse
import pickle

import h5py

from cnn_data_functions import get_partition_indices

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--data", default = "data/data_AB_NN.hdf5,data/data_BA_NN.hdf5,data/data_bi_NN.hdf5")

    parser.add_argument("--val_prop", default = "0.1")
    parser.add_argument("--test_prop", default = "0.1")

    parser.add_argument("--ofile", default = "indices.pkl")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    testProp = float(args.test_prop)
    valProp = float(args.val_prop)

    ifiles = [h5py.File(u, 'r') for u in args.data.split(',')]

    indices = dict()
    keys = ['train', 'test', 'val']

    for key in keys:
        indices[key] = []

    for ifile in ifiles:
        _ = get_partition_indices(list(ifile.keys()), testProp, valProp)

        for key in _.keys():
            indices[key].append(_[key])

    pickle.dump(indices, open(args.ofile, 'wb'))

if __name__ == '__main__':
    main()