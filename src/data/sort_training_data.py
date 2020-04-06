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
    parser.add_argument("--sort_windows", action = "store_true")

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
            print(key)

            logging.debug('{0}: working on key {1} of {2}'.format(comm.rank, ix + 1, len(keys)))

            X_batch = np.array(ifile[key + '/x_0'])
            y_batch = np.array(ifile[key + '/y'])

            if args.sort_windows:
                X_window_batch = np.array(ifile[key + '/x_windows'])
                y_window_batch = np.array(ifile[key + '/y_windows'])

                X_window_new_batch = []
                y_window_new_batch = []

                indices = []

                for k in range(len(X_window_batch)):
                    X_window = []
                    y_window = []
                    ix = []

                    for j in range(X_window_batch.shape[1]):
                        X = copy.copy(X_window_batch[k, j, :, :, 0])
                        y = copy.copy(y_window_batch[k, j, :, :, 0])

                        X, y, i1, i2 = sort_XY(X, y, config, return_indices = True)

                        if args.two_channel:
                            _ = np.zeros((X.shape[0] // 2, X.shape[1], 2))
                            _[:, :, 0] = X[:X.shape[0] // 2, :]
                            _[:, :, 1] = X[X.shape[0] // 2:, :]

                            X = copy.copy(_)

                            y = add_channel(y[y.shape[0] // 2:, :])

                        X_window.append(X)
                        y_window.append(y)
                        ix.append(i2)

                    X_window_new_batch.append(np.array(X_window, dtype = np.uint8))
                    y_window_new_batch.append(np.array(y_window, dtype = np.uint8))
                    indices.append(np.array(ix, dtype = np.uint8))



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

            if args.sort_windows:
                comm.send([np.array(X_new_batch, dtype=np.uint8), np.array(y_new_batch, dtype=np.uint8),
                           np.array(X_window_new_batch, dtype = np.uint8), np.array(y_window_new_batch, dtype = np.uint8), np.array(indices, dtype = np.uint8)], dest=0)
            else:
                comm.send([np.array(X_new_batch, dtype = np.uint8), np.array(y_new_batch, dtype = np.uint8)], dest=0)

    else:
        ofile = h5py.File(args.ofile, 'w')

        n_received = 0

        while n_received < len(keys):
            if not args.sort_windows:
                X_batch, y_batch = comm.recv(source = MPI.ANY_SOURCE)
            else:
                X_batch, y_batch, X_window_batch, y_window_batch, indices = comm.recv(source = MPI.ANY_SOURCE)

                ofile.create_dataset(str(n_received) + '/x_windows', data=X_window_batch, compression='lzf')
                ofile.create_dataset(str(n_received) + '/y_windows', data=y_window_batch, compression='lzf')
                ofile.create_dataset(str(n_received) + '/indices', data=indices, compression='lzf')

            ofile.create_dataset(str(n_received) + '/x_0', data=X_batch, compression='lzf')
            ofile.create_dataset(str(n_received) + '/y', data=y_batch, compression='lzf')

            n_received += 1

        ofile.close()

if __name__ == '__main__':
    main()