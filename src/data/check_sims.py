import os
import numpy as np
import logging, argparse

from data_functions import *

import h5py

import matplotlib.pyplot as plt

from calc_stats_ms import *

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    ms_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if 'ms.gz' in u])
    log_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if 'log.gz' in u])
    out_files = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if '.out' in u])

    for ix in range(len(ms_files)):
        ms = ms_files[ix]
        log = log_files[ix]

        print(ix)
        try:
            X_data, P, itarget, iintrog_reg = load_data_ghost(ms, log, 128, 200)
        except:
            print(ms)
            print(log)

if __name__ == '__main__':
    main()

