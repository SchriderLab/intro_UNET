import logging, argparse
import matplotlib.pyplot as plt

import numpy as np

import os
import pickle
import itertools

import pandas as pd

losses = ['binary_crossentropy', 'dice_coef', 'mixed']

def get_params(ifile, model_names):
    ifile = ifile.split('.')[0]

    poss = itertools.product(model_names, losses)

    for i, j in poss:

        if (i.lower() in ifile.lower()) and (j.lower() in ifile.lower()):
            return i, j, int(ifile.lower().replace(i.lower(), '').replace(j.lower(), '').replace('_', ''))


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

    parser.add_argument("--metric", default = "val_binary_crossentropy")
    parser.add_argument("--ofile", default = "evaluation_metrics.csv")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    model_names = sorted(os.listdir(args.idir))

    result = dict()
    result[args.metric] = list()
    result['model'] = list()
    result['loss'] = list()
    result['batch size'] = list()

    for model in model_names:
        idir = os.path.join(args.idir, model)

        ifiles = os.listdir(idir)

        _ = []

        for ifile in ifiles:

            model, loss, batch_size = get_params(ifile, model_names)

            history = pickle.load(open(os.path.join(idir, ifile), 'rb'))

            result[args.metric].append(np.min(history[args.metric]))
            result['model'].append(model)
            result['loss'].append(loss)
            result['batch size'].append(batch_size)

    ix = np.argmin(result[args.metric])

    print(result['model'][ix])
    print(result['loss'][ix])
    print(result['batch size'][ix])
    print(result[args.metric][ix])

    df = pd.DataFrame(result)
    df.to_csv(args.ofile, header = True, index = False)

    poss = itertools.product(losses, model_names)

    fig, axes = plt.subplots(ncols = 6, nrows = 3, sharey = True)

    axes[0, 0].set_ylabel('binary crossentropy')
    axes[1, 0].set_ylabel('dice coef')
    axes[2, 0].set_ylabel('mixed loss')

    for k in range(len(model_names)):
        axes[0, k].set_title(model_names[k])

    for p in poss:
        i = losses.index(p[0])
        j = model_names.index(p[1])

        df_ = df.loc[(df['loss'] == p[0]) & (df['model'] == p[1])].sort_values('batch size')

        axes[i, j].boxplot(df_[args.metric])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()