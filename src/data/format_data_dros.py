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
    parser.add_argument("--idir", default = "/pine/scr/d/s/dschride/data/popGenCnn/introgression/drosophila/msmodifiedSims/")

    # direction must be mig12, mig21, or noMig
    parser.add_argument("--direction", default = "mig12")

    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--batch_size", default = "4")

    parser.add_argument("--no_y", action="store_true")
    parser.add_argument("--no_channel", action="store_true")

    parser.add_argument("--up_sample", action = "store_true")
    parser.add_argument("--pop_size", default = "32")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if args.direction in u]

    ofile = h5py.File(args.ofile, 'w')

    X = []
    Y = []

    batch_size = int(args.batch_size)
    counter = 0

    for idir in idirs:
        logging.debug('root: working on directory {0}'.format(idir))

        ms_file = os.path.join(idir, '{0}.msOut'.format(args.direction))
        anc_file = os.path.join(idir, 'out.anc')

        x, y = load_data_dros(ms_file, anc_file, up_sample = args.up_sample, up_sample_pop_size = int(args.pop_size))

        X.extend(x)
        Y.extend(y)

        while len(X) > batch_size:
            if not args.no_channel:
                x_data = add_channel(np.array(X[-batch_size:], dtype=np.uint8))
                y_data = add_channel(np.array(Y[-batch_size:], dtype=np.uint8))
            else:
                x_data = np.array(X[-batch_size:], dtype=np.uint8)
                y_data = np.array(Y[-batch_size:], dtype=np.uint8)

            ofile.create_dataset('{0}/x_0'.format(counter), data=x_data, compression='lzf')

            if not args.no_y:
                ofile.create_dataset('{0}/y'.format(counter), data=y_data, compression='lzf')

            del X[-batch_size:]
            del Y[-batch_size:]

            counter += 1

            ofile.flush()

    logging.debug('root: closing file')
    ofile.close()

if __name__ == '__main__':
    main()