import os
import numpy as np
import logging, argparse

from data_functions import *

import h5py

def parse_args():
    # Argument Parser    
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "/proj/dschridelab/introgression_data/sims_raw_v1")
    parser.add_argument("--ofile", default = "/proj/dscridelab/introgression_data/data_v1.0.hdf5")

    parser.add_argument("--batch_size", default = "8")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    ms_files = sorted([u for u in os.listdir(args.idir) if 'ms' in u])
    log_files = sorted([u for u in os.listdir(args.idir) if 'log' in u])

    batch_size = int(args.batch_size)

    ofile = h5py.File(args.ifile)

    X = []
    y = []

    introg = {}

    counter = 0

    for ix in range(len(ms_files)):
        ms = ms_files[ix]
        log = log_files[ix]

        x, ipos, itarget, iintrog_reg = load_data(ms, log, 128, 48)

        X.append(x)
        y.append(itarget)

        for uu in sorted(iintrog_reg):
            introg[kdx] = iintrog_reg[uu]
            kdx+=1

        if len(X) == batch_size:
            ofile.create_dataset('{0}/x_0'.format(counter), data = np.array(X, dtype = np.uint8), compression = 'lzf')
            ofile.create_dataset('{0}/y'.format(counter), data = np.array(y, dtype = np.uint8), compression = 'lzf')

            X = []
            y = []

            counter += 1

    ofile.close()

        
