import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import logging, argparse

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action = "store_true")
    parser.add_argument("--ifile", default = "data_AB_evals.csv")

    args = parser.parse_args()

    if args.verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    df = pd.read_csv(args.ifile, index_col = False)

    mp1 = np.array(df['mig_p1'])
    mp2 = np.array(df['mig_p2'])
    acc = np.array(df['accuracy'])

    ix = np.intersect1d(np.where(mp1 != 0), np.where(mp2 != 0))
    ix = list(ix)

    mp1 = mp1[ix]
    mp2 = mp2[ix]
    acc = acc[ix]

    mp1_bins = np.linspace(np.min(mp1), np.max(mp1) + 0.00001, 30)
    mp2_bins = np.linspace(np.min(mp2), np.max(mp2) + 0.00001, 30)

    Z = np.zeros((len(mp1_bins) - 1, len(mp1_bins) - 1))
    counts = np.zeros((len(mp2_bins) - 1, len(mp2_bins) - 1))

    for k in range(len(mp1)):

        i = np.digitize(mp1[k], mp1_bins) - 1
        j = np.digitize(mp2[k], mp2_bins) - 1

        Z[i, j] += acc[k]
        counts[i,j] += 1.

    fig, ax = plt.subplots(figsize=(6,6))

    im = ax.imshow(np.flip(Z / counts, axis = 0), extent = (np.min(mp2_bins), np.max(mp2_bins), np.min(mp1_bins), np.max(mp1_bins)))

    ax.set_xlabel('migration probability (1 -> 2)')
    ax.set_ylabel('migration probability (2 -> 1)')

    ax.set_title('accuracy')

    plt.colorbar(im , ax = ax)
    plt.show()

if __name__ == '__main__':
    main()
