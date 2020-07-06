import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform

import random
import h5py
import numpy as np

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--ofile", default = "None")

    parser.add_argument("--sample", default = "0.01")

    parser.add_argument("--odir", default = "None")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.mkdir(args.odir)
            logging.debug('root: made output directory {0}'.format(args.odir))
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args

def main():
    args = parse_args()

    ifile = h5py.File(args.ifile, 'r')

    keys = ifile.keys()

    ofile = open(args.ofile, 'w')

    for key in ifile.keys():
        features = np.array(ifile[key]['features'])

        features = features.reshape((features.shape[0]*features.shape[1]*features.shape[2], features.shape[3]))

        indices = list(range(len(features)))

        indices = random.sample(indices, int(np.ceil(len(indices)*float(args.sample))))

        for ix in indices:
            ofile.write(','.join(list(map(str, features[ix]))) + '\n')

    ofile.close()


if __name__ == '__main__':
    main()