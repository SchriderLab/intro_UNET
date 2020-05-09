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

    X_windows = []
    Y_windows = []

    batch_size = int(args.batch_size)
    counter = 0

    for idir in idirs:
        logging.debug('root: working on directory {0}'.format(idir))

        ms_file = os.path.join(idir, '{0}.msOut'.format(args.direction))
        anc_file = os.path.join(idir, 'out.anc')

        x, y = load_data_dros(ms_file, anc_file, n_sites = None, up_sample = args.up_sample, up_sample_pop_size = int(args.pop_size))

        for k in range(len(x)):

            ipos = list(range(x[k].shape[1]))
            windows, middle_indices = get_windows_snps(x[k], ipos, N = 64)

            X.append(x[k][:, middle_indices])
            Y.append(y[k][:, middle_indices])

            x_w = []
            y_w = []

            for w in windows:
                x_w.append(x[k][:, w])
                y_w.append(y[k][:, w])

            X_windows.append(np.array(x_w, dtype = np.uint8))
            Y_windows.append(np.array(y_w, dtype = np.uint8))


        while len(X) > batch_size:
            if not args.no_channel:
                x_data = add_channel(np.array(X[-batch_size:], dtype=np.uint8))
                y_data = add_channel(np.array(y[-batch_size:], dtype=np.uint8))

                x_window_data = add_channel(np.array(X_windows[-batch_size:], dtype = np.uint8))
                y_window_data = add_channel(np.array(Y_windows[-batch_size:], dtype = np.uint8))
            else:
                x_data = np.array(X[-batch_size:], dtype=np.uint8)
                y_data = np.array(y[-batch_size:], dtype=np.uint8)

                x_window_data = np.array(X_windows[-batch_size:], dtype=np.uint8)
                y_window_data = np.array(Y_windows[-batch_size:], dtype=np.uint8)

            print(x_data.shape, y_data.shape)

            ofile.create_dataset('{0}/x_0'.format(counter), data=x_data, compression='lzf')
            ofile.create_dataset('{0}/x_windows'.format(counter), data = x_window_data, compression = 'lzf')

            if not args.no_y:
                ofile.create_dataset('{0}/y'.format(counter), data=y_data, compression='lzf')
                ofile.create_dataset('{0}/y_windows'.format(counter), data=y_window_data, compression = 'lzf')

            del X[-batch_size:]
            del y[-batch_size:]

            counter += 1

            ofile.flush()

    logging.debug('root: closing file')
    ofile.close()

if __name__ == '__main__':
    main()