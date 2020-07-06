import os
import logging, argparse
import h5py

import numpy as np
import random

import matplotlib.pyplot as plt
import cv2
import sys

def random_choice(keys, batches):
    return (random.choice(keys), random.choice(batches))

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")

    parser.add_argument("--ifile", default = "archie_data_10e6.hdf5")
    parser.add_argument("--ofile", default = "archie_200_sample.hdf5")

    parser.add_argument("--n_samples", default = "1000")
    parser.add_argument("--batch_size", default = "4")

    parser.add_argument("--no_replacement", action = "store_true")

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
    batches = list(range(4))

    n_samples = int(args.n_samples)
    batch_size = int(args.batch_size)

    X = []
    Y = []

    counter = 0

    chosen = []

    todo = dict()

    while len(chosen) < n_samples:
        key, batch = random_choice(keys, batches)

        if args.no_replacement:
            while (key, batch) in chosen:
                key, batch = random_choice(keys, batches)

        if key in todo.keys():
            todo[key].append(batch)
        else:
            todo[key] = [batch]

        chosen.append((key, batch))

    print(len(chosen))

    for key in todo.keys():

        x = np.array(ifile['{0}/x_0'.format(key)])
        y = np.array(ifile['{0}/y'.format(key)])

        for batch in todo[key]:
            X.append(x[batch])
            Y.append(y[batch])

        while len(X) >= batch_size:
            ofile.create_dataset('{0}/x_0'.format(counter),
                                 data=np.array(X[-batch_size:], dtype=np.uint8), compression='lzf')
            ofile.create_dataset('{0}/y'.format(counter),
                                 data=np.array(Y[-batch_size:], dtype=np.uint8), compression='lzf')

            del X[-batch_size:]
            del Y[-batch_size:]

            counter += 1
            print(counter)

    ofile.close()

if __name__ == '__main__':
    main()