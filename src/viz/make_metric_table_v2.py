import logging, argparse
import matplotlib.pyplot as plt

import numpy as np

import os
import pickle
import itertools

import pandas as pd

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

    parser.add_argument("--ofile", default = "evaluation_metrics.csv")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    print('stuff')
    return args

def main():
    args = parse_args()

    sortings = os.listdir(args.idir)

    print(sortings)

    result = dict()
    result['model'] = list()
    result['batch size'] = list()
    result['sorting'] = list()
    result['validation loss'] = list()
    result['validation accuracy'] = list()

    for sorting in sortings:
        idir = os.path.join(args.idir, sorting)
        print(idir)

        ifiles = os.listdir(idir)


        for ifile in ifiles:
            model = ifile.split('.')[0].split('_')[-1]
            batch_size = ifile.split('.')[0].split('_')[1]

            history = pickle.load(open(os.path.join(idir, ifile), 'rb'))

            print(history.keys())

            val_loss = history['val_loss']
            acc = history['acc']

            vl = np.min(val_loss)
            acc = acc[np.argmin(val_loss)]

            result['model'].append(model)
            result['batch size'].append(batch_size)
            result['sorting'].append(sorting)
            result['validation loss'].append(vl)
            result['validation accuracy'].append(acc)

    df = pd.DataFrame(result)
    df.to_csv(args.ofile, header = True, index = False)


if __name__ == '__main__':
    main()