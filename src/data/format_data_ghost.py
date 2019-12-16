import os
import numpy as np
import logging, argparse

from data_functions import *

import h5py

import matplotlib.pyplot as plt

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")

    parser.add_argument("--ms", default = "None")
    parser.add_argument("--log", default = "None")
    parser.add_argument("--otag", default = "None")

    parser.add_argument("--format_mode", default = "sort_NN")

    parser.add_argument("--n_individuals", default = "64")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    ms = args.ms
    log = args.log

    X, P, itarget, iintrog_reg = load_data_ghost(ms, log, 128, int(args.n_individuals))

    counter = 0

    for k in range(len(X)):
        x = X[k]
        ipos = P[k]
        y = itarget[k]

        windows, middle_indices = get_windows(x, ipos)

        odir = '{0}_{1:04d}_windows'.format(args.otag, counter)
        os.mkdir(odir)

        for ix in range(len(windows)):
            # make 4 files for each window
            w = windows[ix]

            x_ = x[:,w]
            ipos_ = ipos[w]
            y_ = y[:,w]

            write_snp_file(list(map(int, ipos_)), os.path.join(odir, '{0:04d}.snp'.format(ix)))
            write_binary_matrix(x_[:len(x_) // 2], os.path.join(odir, '{0:04d}.ref.geno'.format(ix)))
            write_binary_matrix(x_[len(x_) // 2:], os.path.join(odir, '{0:04d}.ADMIXED.geno'.format(ix)))
            write_binary_matrix(y_[len(x_) // 2:], os.path.join(odir, '{0:04d}.ADMIXED.anc'.format(ix)))

        np.savez('{0}.npz'.format('{0}_{1:04d}'.format(args.otag, counter)), X = x[:, middle_indices], Y = y[:, middle_indices])

        counter += 1

if __name__ == '__main__':
    main()

