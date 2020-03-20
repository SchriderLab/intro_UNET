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

    mp = np.array(df['mig_p1'])
    mp1 = np.array(df['mig_p2'])
    mt = np.array(df['mig_time'])
    acc = np.array(df['accuracy'])



    ix = np.intersect1d(np.where(mp == 0)[0], np.where(mp1 != 0)[0])
    ix = list(ix)
    print(len(ix))

    mp = mp1[ix]
    mt = mt[ix]
    acc = acc[ix]

    print(np.mean(acc))

    mt_bins = np.linspace(np.min(mt), np.max(mt) + 1, 30)
    mp_bins = np.linspace(np.min(mp), np.max(mp) + 0.0001, 30)

    Z = np.zeros((len(mt_bins) - 1, len(mp_bins) - 1))
    counts = np.zeros((len(mt_bins) - 1, len(mp_bins) - 1))

    for k in range(len(mp)):

        i = np.digitize(mt[k], mt_bins) - 1
        j = np.digitize(mp[k], mp_bins) - 1

        Z[i, j] += acc[k]
        counts[i,j] += 1.

    fig, ax = plt.subplots(figsize=(6,6))

    im = ax.imshow(np.flip(Z / counts, axis = 0), extent = (np.min(mp), np.max(mp), np.min(mt), np.max(mt)), aspect = 'auto')

    ax.set_xlabel('migration probability')
    ax.set_ylabel('migration time')

    ax.set_title('accuracy')

    plt.colorbar(im , ax = ax)
    plt.show()

if __name__ == '__main__':
    main()
