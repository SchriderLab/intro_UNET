import os
import numpy as np
import logging, argparse

from data_functions import *

import h5py
import copy

import configparser
from mpi4py import MPI

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--format_config", default = "None")

    parser.add_argument("--two_channel", action = "store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    # configure MPI
    comm = MPI.COMM_WORLD

    ifile = h5py.File(args.ifile, 'r')

    keys = list(ifile.keys())

    config = configparser.ConfigParser()
    config.read(args.format_config)

    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(keys), comm.size - 1):
            key = keys[ix]

            logging.debug('{0}: working on key {1} of {2}'.format(comm.rank, ix + 1, len(keys)))

            X_batch = np.array(ifile[key + '/x_0'])
            y_batch = np.array(ifile[key + '/y'])

            X_new_batch = []
            y_new_batch = []

            for k in range(len(X_batch)):
                X = copy.copy(X_batch[k,:,:,0])
                y = copy.copy(y_batch[k,:,:,0])

                X, y = sort_XY(X, y, config)

                if args.two_channel:
                    _ = np.zeros((X.shape[0] // 2, X.shape[1], 2))
                    _[:, :, 0] = X[:X.shape[0] // 2, :]
                    _[:, :, 1] = X[X.shape[0] // 2:, :]

                    X = copy.copy(_)

                    y = add_channel(y[y.shape[0] // 2:, :])

                X_new_batch.append(X)
                y_new_batch.append(y)

            comm.send([np.array(X_new_batch, dtype = np.uint8), np.array(y_new_batch, dtype = np.uint8)], dest=0)

    else:
        ofile = h5py.File(args.ofile, 'w')

        n_received = 0

        while n_received < len(keys):
            X_batch, y_batch = comm.recv(source = MPI.ANY_SOURCE)

            ofile.create_dataset(str(n_received) + '/x_0', data = X_batch, compression = 'lzf')
            ofile.create_dataset(str(n_received) + '/y', data = y_batch, compression = 'lzf')

            n_received += 1

        ofile.close()

if __name__ == '__main__':
    main()