import logging, argparse
import matplotlib.pyplot as plt

import numpy as np

import os
import pickle
import itertools

import pandas as pd

from itertools import product

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")

    parser.add_argument("--metrics", default = "metrics/")
    parser.add_argument("--evals", default = "perturbation_evals")

    parser.add_argument("--ofile", default = "perturbation_experiment.csv")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    evals = sorted(os.listdir(args.evals))
    coefficients = [(16, 0.1), (16, 0.25), (16, 0.75), (1, 0.1), (1, 0.25), (1, 0.75), (4, 0.1), (4, 0.25), (4, 0.75)]

    metrics = sorted(os.listdir(args.metrics))

    result = dict()
    result['split time coefficient'] = []
    result['migration time coefficient'] = []
    result['original accuracy'] = []
    result['accuracy'] = []

    for ix in range(len(evals)):
        loss, acc, dc = np.loadtxt(os.path.join(args.evals, evals[ix]))
        ret = pickle.load(open(os.path.join(args.metrics, metrics[ix]), 'rb'))

        st, mt = coefficients[ix]

        result['split time coefficient'].append(st)
        result['migration time coefficient'].append(mt)
        result['original accuracy'].append(acc)
        result['accuracy'].append(ret['accuracy'])

    df = pd.DataFrame(result)
    df.to_csv(args.ofile, header=True, index=False)

if __name__ == '__main__':
    main()