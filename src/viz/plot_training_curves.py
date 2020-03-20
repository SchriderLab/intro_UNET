import logging, argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle

import os

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "training_output/bs_64")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    ifiles = [u for u in os.listdir(args.idir) if u.split('.')[-1] == 'pkl']

    histories = [pickle.load(open(os.path.join(args.idir, u), 'rb')) for u in ifiles]

    plt.rc('font',family='Arial',size=11) # set to Arial/Helvetica
    plt.rcParams.update({'figure.autolayout':True})
    fig = plt.figure(figsize=(8,8),dpi=100)

    ax = fig.add_subplot(1, 1, 1)

    minimums = dict()

    for k in range(len(ifiles)):
        history = histories[k]

        ax.plot(history['val_loss'], label = ifiles[k].split('.')[0])
        minimums[ifiles[k].split('.')[0]] = np.min(history['val_loss'])

    key = min(minimums, key=minimums.get)
    print(key)
    print(minimums[key])

    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('binary crossentropy')

    plt.show()

if __name__ == '__main__':
    main()
