import os
import logging, argparse
import h5py

import numpy as np
import random

import matplotlib.pyplot as plt
import cv2
import sys

def random_choice(keys, batches, windows):
    return (random.choice(keys), random.choice(batches), random.choice(windows))

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")

    parser.add_argument("--ifile", default = "archie_data_all_windows.hdf5")
    parser.add_argument("--ofile", default = "archie_200_sample.hdf5")

    parser.add_argument("--n_samples", default = "1000")
    parser.add_argument("--batch_size", default = "4")

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
    batches = list(range(8))
    windows = list(range(255))

    n_samples = int(args.n_samples)
    batch_size = int(args.batch_size)

    X = []
    Y = []

    counter = 0

    chosen = []

    while len(chosen) < n_samples:
        key, batch, window = random_choice(keys, batches, windows)
        while '{0}_{1}_{2}'.format(key, batch, window) in chosen:
            key, batch, window = random_choice(keys, batches, windows)

        chosen.append((key, batch, window))

    todo = dict()

    keys = [u[0] for u in chosen]
    for key in set(keys):
        todo[key] = []
        for c in chosen:
            if c[0] == key:
                todo[key].append((c[1], c[2]))

    for key in todo.keys():

        x = np.array(ifile['{0}/x_windows'.format(key)])
        y = np.array(ifile['{0}/y_windows'.format(key)])

        for batch, window in todo[key]:
            X.append(x[batch, window])
            Y.append(y[batch, window])

        while len(X) >= batch_size:
            ofile.create_dataset('{0}/x_0'.format(counter),
                                 data=np.array(X[-batch_size:], dtype=np.uint8), compression='lzf')
            ofile.create_dataset('{0}/y'.format(counter),
                                 data=np.array(Y[-batch_size:], dtype=np.uint8), compression='lzf')

            del X[-batch_size:]
            del Y[-batch_size:]

            counter += 1

    ofile.close()

if __name__ == '__main__':
    main()