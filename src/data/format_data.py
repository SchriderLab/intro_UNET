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
    parser.add_argument("--idir", default = "/proj/dschridelab/introgression_data/sims_raw_v1")
    parser.add_argument("--ofile", default = "None")

    parser.add_argument("--format_mode", default = "sort_NN")

    parser.add_argument("--batch_size", default = "8")

    parser.add_argument("--n_individuals", default = "48")
    parser.add_argument("--n_sites", default = "128")

    parser.add_argument("--down_size", default = "0")

    parser.add_argument("--no_y", action = "store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    ms_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if 'ms.gz' in u])
    log_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if 'log.gz' in u])
    out_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if '.out' in u])

    batch_size = int(args.batch_size)

    ofile = h5py.File(args.ofile, 'w')

    X = []
    y = []
    params = []

    introg = {}

    counter = 0
    kdx = 0

    for ix in range(len(ms_files)):
        ms = ms_files[ix]
        log = log_files[ix]
        out = out_files[ix]

        x, ipos, itarget, iintrog_reg = load_data(ms, log, int(args.n_sites), int(args.n_individuals))
        p = get_params(out)

        params.extend(p)

        if args.format_mode == 'None':
            X.extend(x)
            y.extend(itarget)
        elif args.format_mode == 'sort_NN':
            for k in range(len(x)):
                x_, indices = sort_NN(x[k])
                itarget_ = itarget[k][indices]

                X.append(x_)
                y.append(itarget_)
        elif args.format_mode == 'min_match_sorted':
            for k in range(len(x)):
                x_, indices = sort_cdist(x[k], opt = 'min', sort_pop = True)
                itarget_ = itarget[k][indices]

                X.append(x_)
                y.append(itarget_)

        elif args.format_mode == 'max_match_sorted':
            for k in range(len(x)):
                x_, indices = sort_cdist(x[k], opt = 'max', sort_pop = True)
                itarget_ = itarget[k][indices]

                X.append(x_)
                y.append(itarget_)

        elif args.format_mode == 'min_match':
            for k in range(len(x)):
                x_, indices = sort_cdist(x[k], opt='min', sort_pop=False)

                itarget_ = itarget[k][indices]

                X.append(x_)
                y.append(itarget_)
        elif args.format_mode == 'max_match':
            for k in range(len(x)):
                x_, indices = sort_cdist(x[k], opt='max', sort_pop=False)
                itarget_ = itarget[k][indices]

                X.append(x_)
                y.append(itarget_)

        for uu in sorted(iintrog_reg):
            introg[kdx] = iintrog_reg[uu]
            kdx+=1

        while len(X) >= batch_size:
            logging.debug('root: making batch {0}'.format(counter))
            
            ofile.create_dataset('{0}/x_0'.format(counter), data = add_channel(np.array(X[-batch_size:], dtype = np.uint8)[:,int(args.down_size):,:]), compression = 'lzf')
            if not args.no_y:
                ofile.create_dataset('{0}/y'.format(counter), data = add_channel(np.array(y[-batch_size:], dtype = np.uint8)[:,int(args.down_size):,:]), compression = 'lzf')
            ofile.create_dataset('{0}/params'.format(counter), data = np.array(params[-batch_size:]), dtype = np.float32, compression = 'lzf')

            del X[-batch_size:]
            del y[-batch_size:]
            del params[-batch_size:]

            counter += 1

    ofile.close()

if __name__ == '__main__':
    main()

        
