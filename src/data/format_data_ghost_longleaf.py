import os
import numpy as np
import logging, argparse

from data_functions import *

import h5py

import matplotlib.pyplot as plt

from calc_stats_ms import *

def get_feature_vector(mutation_positions, genotypes, ref_geno, arch):
    n_samples = len(genotypes[0])

    n_sites = 500

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

    parser.add_argument("--ms", default = "None")
    parser.add_argument("--log", default = "None")
    parser.add_argument("--out", default = "None")

    parser.add_argument("--format_mode", default = "sort_NN")

    parser.add_argument("--n_individuals", default = "64")
    parser.add_argument("--n_per_file", default = "100")

    parser.add_argument("--batch_size", default="8")

    parser.add_argument("--ofile", default = "archie_data.hdf5")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    ofile = h5py.File(args.ofile, 'w')
    counter = 0

    ms = args.ms
    log = args.log
    out = args.out

    batch_size = int(args.batch_size)

    p = get_params_ghost(out)

    X_data, P, itarget, iintrog_reg = load_data_ghost(ms, log, 128, int(args.n_individuals))

    X = []
    Y = []
    features = []
    positions = []
    params = []

    for k in range(len(X_data)):
        x = X_data[k]
        ipos = P[k]
        y = itarget[k]

        windows, middle_indices = get_windows(x, ipos)
        middle_pos = list(ipos[middle_indices])

        fs = []
        ps = []

        for i in range(len(windows)):
            # make 4 files for each window
            w = windows[i]

            x_ = x[:, w]
            ipos_ = ipos[w]
            y_ = y[:, w]

            pos = np.zeros(len(middle_pos), dtype=np.uint8)

            for j in range(len(pos)):
                if middle_pos[j] in ipos_:
                    pos[j] = 1

            genotypes = list(map(list, list(x_[len(x) // 2:, :].T.astype(np.uint8))))
            ref_geno = list(map(list, list(x_[:len(x) // 2, :].T.astype(np.uint8))))
            mutation_positions = list(ipos_.astype(np.int32))
            arch = list(map(list, list(y_[len(x) // 2:, :].T.astype(np.uint8).astype(str))))

            f = get_feature_vector(mutation_positions, genotypes, ref_geno, arch)

            fs.append(f)
            ps.append(pos)

        X.append(x[:, middle_indices])
        Y.append(y[:, middle_indices])
        features.append(np.array(fs, dtype = np.float32))
        positions.append(np.array(ps, dtype = np.float32))
        params.append(p[k])

        while len(X) >= batch_size:
            ofile.create_dataset('{0}/x_0'.format(counter),
                                 data=add_channel(np.array(X[-batch_size:], dtype=np.uint8)), compression='lzf')
            ofile.create_dataset('{0}/y'.format(counter),
                                 data=add_channel(np.array(Y[-batch_size:], dtype=np.uint8)), compression='lzf')

            f, p = pad_matrices(features[-batch_size:], positions[-batch_size:])

            ofile.create_dataset('{0}/features'.format(counter), data=f, compression='lzf')
            ofile.create_dataset('{0}/positions'.format(counter), data=p, compression='lzf')
            ofile.create_dataset('{0}/params'.format(counter), data=np.array(params[-batch_size:]),
                                 dtype=np.float32, compression='lzf')

            del X[-batch_size:]
            del Y[-batch_size:]
            del features[-batch_size:]
            del positions[-batch_size:]
            del params[-batch_size:]

            counter += 1

    ofile.close()

if __name__ == '__main__':
    main()

