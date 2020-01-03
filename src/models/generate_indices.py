import os
import numpy as np
import logging, argparse
import pickle

import h5py

def get_partition_indices(batches, testProp, valProp):
    ret = dict()

    n_batches = len(batches)

    n_train = int(np.floor((1 - (testProp + valProp)) * n_batches))
    n_val = int(np.floor(valProp * n_batches))
    n_test = int(np.floor(testProp * n_batches))

    ret['train'] = np.random.choice(batches, n_train, replace=False)
    batches = list(set(batches).difference(ret['train']))

    ret['val'] = np.random.choice(batches, n_val, replace=False)
    batches = list(set(batches).difference(ret['val']))

    ret['test'] = np.random.choice(batches, n_test, replace=False)
    batches = list(set(batches).difference(ret['test']))

    return ret

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