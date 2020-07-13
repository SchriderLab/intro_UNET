import os
import numpy as np
import logging, argparse

from data_functions import *

import h5py

import matplotlib.pyplot as plt

from calc_stats_ms import *
from mpi4py import MPI

import copy

def get_feature_vector(mutation_positions, genotypes, ref_geno, arch):
    n_samples = len(genotypes[0])

    n_sites = 50000

    ## set up S* stuff -- remove mutations found in reference set
    t_ref = list(map(list, zip(*ref_geno)))
    t_geno = list(map(list, zip(*genotypes)))
    pos_to_remove = set()  # contains indexes to remove
    s_star_haps = []
    for idx, hap in enumerate(t_ref):
        for jdx, site in enumerate(hap):
            if site == 1:
                pos_to_remove.add(jdx)

    for idx, hap in enumerate(t_geno):
        s_star_haps.append([v for i, v in enumerate(hap) if i not in pos_to_remove])

    ret = []

    # individual level
    for focal_idx in range(0, n_samples):
        calc_N_ton = N_ton(genotypes, n_samples, focal_idx)
        dist = distance_vector(genotypes, focal_idx)
        min_d = [min_dist_ref(genotypes, ref_geno, focal_idx)]
        ss = [s_star_ind(np.array(s_star_haps), np.array(mutation_positions), focal_idx)]
        n_priv = [num_private(np.array(s_star_haps), focal_idx)]
        focal_arch = [row[focal_idx] for row in arch]
        lab = label(focal_arch, mutation_positions, n_sites, 0.7, 0.3)

        output = calc_N_ton + dist + min_d + ss + n_priv + lab
        ret.append(output)

    return np.array(ret, dtype = np.float32)

def pad_matrices(features, positions):
    max_window_size = max([u.shape[0] for u in features])

    features = [np.pad(u, ((0, max_window_size - u.shape[0]), (0, 0), (0, 0)), 'constant') for u in features]
    positions = [np.pad(u, ((0, max_window_size - u.shape[0]), (0, 0)), 'constant') for u in positions]

    return np.array(features, dtype = np.float32), np.array(positions, dtype = np.uint8)


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

    parser.add_argument("--format_mode", default = "sort_NN")

    parser.add_argument("--n_individuals", default = "200")
    parser.add_argument("--n_per_file", default = "8")

    parser.add_argument("--batch_size", default="8")

    parser.add_argument("--ofile", default = "archie_data.hdf5")

    parser.add_argument("--two_channel", action = "store_true")
    parser.add_argument("--zero_check", action = "store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    # configure MPI
    comm = MPI.COMM_WORLD

    args = parse_args()

    ms_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if 'ms.gz' in u])
    log_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if 'log.gz' in u])
    out_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if '.out' in u])

    n_sims = int(args.n_per_file)*len(ms_files)

    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(ms_files), comm.size - 1):
            ms = ms_files[ix]
            log = log_files[ix]
            out = out_files[ix]

            p = get_params_ghost(out)

            X_data, P, itarget, iintrog_reg = load_data_ghost(ms, log, 128, int(args.n_individuals))

            for k in range(len(X_data)):
                logging.debug('{0}: working on file {1}, dataset {2}'.format(comm.rank, ix, k))

                x = X_data[k]
                ipos = P[k]
                y = itarget[k]

                X = x[:,:128]
                Y = y[:,:128]

                print(X.shape, Y.shape)

                if Y.shape[1] == 128:
                    comm.send([X, Y, p[k]], dest = 0)
                else:
                    comm.send([np.zeros((192, 128), dtype = np.uint8), np.zeros((192, 128), dtype = np.uint8), p[k]], dest = 0)

    else:
        ofile = h5py.File(args.ofile, 'w')

        n_recieved = 0
        batch_size = int(args.batch_size)

        counter = 0

        X = []
        Y = []
        params = []

        while n_recieved < n_sims:
            x, y, param = comm.recv(source = MPI.ANY_SOURCE)

            n_recieved += 1

            if n_recieved % 10 == 0:
                logging.debug('0: recieved {0} simulations'.format(n_recieved))

            if not args.zero_check:
                X.append(x)
                Y.append(y)
                params.append(param)
            else:
                if np.sum(y) > 0:
                    X.append(x)
                    Y.append(y)
                    params.append(param)

            while len(X) >= batch_size:
                x_data = np.array(X[-batch_size:], dtype=np.uint8)
                y_data = np.array(Y[-batch_size:], dtype=np.uint8)

                if len(x_data.shape) == 3:
                    x_data = add_channel(x_data)
                    y_data = add_channel(y_data)

                ofile.create_dataset('{0}/x_0'.format(counter),
                                     data=x_data, compression='lzf')
                ofile.create_dataset('{0}/y'.format(counter),
                                     data=y_data, compression='lzf')

                ofile.create_dataset('{0}/params'.format(counter), data=np.array(params[-batch_size:]),
                                     dtype=np.float32, compression='lzf')

                ofile.flush()

                del X[-batch_size:]
                del Y[-batch_size:]
                del params[-batch_size:]

                counter += 1

        logging.debug('0: closing file...')
        ofile.close()


if __name__ == '__main__':
    main()

# sbatch -p 528_queue -n 512 -t 1-00:00:00 --wrap "mpirun -oversubscribe python3 src/data/format_data_ghost.py --idir /proj/dschridelab/introgression_data/sims_64_10e5_ghost/ --ofile /proj/dschridelab/ddray/archie_64_data.hdf5 --verbose"
# sbatch -p 528_queue -n 512 -t 1-00:00:00 --wrap "mpirun -oversubscribe python3 src/data/format_data_ghost.py --idir /proj/dschridelab/introgression_data/sims_200_10e5_ghost/ --ofile /proj/dschridelab/ddray/archie_200_data.hdf5 --verbose --n_individuals 200 --window_size 50000"