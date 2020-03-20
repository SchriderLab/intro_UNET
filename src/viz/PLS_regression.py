import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import logging, argparse

from sklearn.cross_decomposition import PLSRegression

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

    df = pd.read_csv(args.ifile, index_col=False)

    X = np.array([df['mig_p1'], df['mig_p2'], df['mig_time']]).T
    y = np.array(df['accuracy'])
    y = y.reshape(-1, 1)

    clf = PLSRegression(n_components = 3)
    clf.fit(X, y)

    print(clf.score(X, y))

if __name__ == '__main__':
    main()
