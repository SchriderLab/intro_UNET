import matplotlib.pyplot as plt
import numpy as np
import h5py

import os
import logging, argparse

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/kilgoretrout/intro_UNET/src/data/')

from data_functions import *
import os

import random

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "archie_data_all_windows.hdf5")
    parser.add_argument("--format_mode", default = "different_sortings")

    parser.add_argument("--odir", default = "different_sortings")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.odir):
        os.mkdir(args.odir)
        logging.debug('root: made output directory {0}'.format(args.odir))
    else:
        os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args

def main():
    args = parse_args()

    ifile = h5py.File(args.ifile, 'r')
    keys = sorted(list(map(int, ifile.keys())))

    random.shuffle(keys)

    counter = 0

    for key in keys[:1]:
        X = ifile['{0}/x_0'.format(key)]
        Y = ifile['{0}/y'.format(key)]

        for k in range(len(X)):
            odir = os.path.join(args.odir, '{0:03d}'.format(counter))
            os.mkdir(odir)

            x = X[k,:,:,0]
            y = Y[k,:,:,0]

            plt.rc('font', family='Arial', size=12)  # set to Arial/Helvetica
            plt.rcParams.update({'figure.autolayout': True})
            fig = plt.figure(figsize=(9, 16), dpi=100)

            ax1 = fig.add_subplot(2, 1, 1)
            ax1.get_xaxis().set_visible(False)

            ax1.imshow(x, cmap='gray')

            ax2 = fig.add_subplot(2, 1, 2)

            ax2.imshow(y, cmap='gray')

            plt.savefig(os.path.join(odir, 'no_sorting.eps'))
            plt.close()

            x, indices = sort_NN(x)
            y = y[indices]

            plt.rc('font', family='Arial', size=12)  # set to Arial/Helvetica
            plt.rcParams.update({'figure.autolayout': True})
            fig = plt.figure(figsize=(9, 16), dpi=100)

            ax1 = fig.add_subplot(2, 1, 1)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

            ax1.imshow(x, cmap = 'gray')

            ax2 = fig.add_subplot(2, 1, 2, sharex = ax1)
            ax2.get_yaxis().set_visible(False)

            ax2.imshow(y, cmap = 'gray')

            plt.savefig(os.path.join(odir, 'sort_NN.eps'))
            plt.close()

            x, indices = sort_NN(x, method='max')
            y = y[indices]

            plt.rc('font', family='Arial', size=12)  # set to Arial/Helvetica
            plt.rcParams.update({'figure.autolayout': True})
            fig = plt.figure(figsize=(9, 16), dpi=100)

            ax1 = fig.add_subplot(2, 1, 1)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

            ax1.imshow(x, cmap='gray')

            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
            ax2.get_yaxis().set_visible(False)

            ax2.imshow(y, cmap = 'gray')

            plt.savefig(os.path.join(odir, 'sort_NN_max.eps'))
            plt.close()

            x, indices = sort_cdist(x, opt = 'max', sort_pop = True)
            y = y[indices]

            plt.rc('font', family='Arial', size=12)  # set to Arial/Helvetica
            plt.rcParams.update({'figure.autolayout': True})
            fig = plt.figure(figsize=(9, 16), dpi=100)

            ax1 = fig.add_subplot(2, 1, 1)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

            ax1.imshow(x, cmap='gray')

            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
            ax2.get_yaxis().set_visible(False)

            ax2.imshow(y, cmap = 'gray')

            plt.savefig(os.path.join(odir, 'max_match_sorted.eps'))
            plt.close()

            x, indices = sort_cdist(x, opt='min', sort_pop=True)
            y = y[indices]


            plt.rc('font', family='Arial', size=12)  # set to Arial/Helvetica
            plt.rcParams.update({'figure.autolayout': True})
            fig = plt.figure(figsize=(9, 16), dpi=100)

            ax1 = fig.add_subplot(2, 1, 1)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

            ax1.imshow(x, cmap='gray')

            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
            ax2.get_yaxis().set_visible(False)

            ax2.imshow(y, cmap = 'gray')

            plt.savefig(os.path.join(odir, 'min_match_sorted.eps'))
            plt.close()

            x, indices = sort_cdist(x, opt = 'min', sort_pop = False)
            y = y[indices]

            plt.rc('font', family='Arial', size=12)  # set to Arial/Helvetica
            plt.rcParams.update({'figure.autolayout': True})
            fig = plt.figure(figsize=(9, 16), dpi=100)

            ax1 = fig.add_subplot(2, 1, 1)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

            ax1.imshow(x, cmap='gray')

            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
            ax2.get_yaxis().set_visible(False)

            ax2.imshow(y, cmap = 'gray')

            plt.savefig(os.path.join(odir, 'min_match.eps'))
            plt.close()

            x, indices = sort_cdist(x, opt = 'max', sort_pop = False)
            y = y[indices]

            plt.rc('font', family='Arial', size=12)  # set to Arial/Helvetica
            plt.rcParams.update({'figure.autolayout': True})
            fig = plt.figure(figsize=(9, 16), dpi=100)

            ax1 = fig.add_subplot(2, 1, 1)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

            ax1.imshow(x, cmap='gray')

            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
            ax2.get_yaxis().set_visible(False)

            ax2.imshow(y, cmap = 'gray')

            plt.savefig(os.path.join(odir, 'max_match.eps'))
            plt.close()

            counter += 1



if __name__ == '__main__':
    main()