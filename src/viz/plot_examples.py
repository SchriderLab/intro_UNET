import matplotlib.pyplot as plt
import numpy as np
import h5py

import os
import logging, argparse

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/kilgoretrout/intro_UNET/src/data/')

from data_functions import *

from scipy.spatial.distance import cdist

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "archie_200_data_slim.hdf5")
    parser.add_argument("--format_mode", default = "None")

    parser.add_argument("--odir", default = "None")
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

    counter = 0

    for key in keys[:25]:
        X = ifile['{0}/x_0'.format(key)]
        Y = ifile['{0}/y'.format(key)]

        for k in range(len(X)):
            x = X[k,:,:,0]
            y = Y[k,:,:,0]

            fig, axes = plt.subplots(nrows = 2)

            axes[0].imshow(x, cmap = 'gray')
            axes[1].imshow(y, cmap = 'gray')

            plt.show()
            plt.close()



if __name__ == '__main__':
    main()

