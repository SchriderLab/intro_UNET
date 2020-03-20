import logging, argparse
import matplotlib.pyplot as plt

import numpy as np

import os
import pickle
import itertools

import pandas as pd

from itertools import product

class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.
    It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
    create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
    attributes.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    display_labels : ndarray of shape (n_classes,)
        Display labels for plot.
    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.
    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.
    ax_ : matplotlib Axes
        Axes with confusion matrix.
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, include_values=True, cmap='Blues',
             xticks_rotation='horizontal', values_format=None, ax=None):
        """Plot visualization.
        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is '.2g'.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """
        #check_matplotlib_support("ConfusionMatrixDisplay.plot")
        #import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = '.2g'

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color)

        fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=self.display_labels,
               yticklabels=self.display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--odir", default = "None")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.odir):
        os.mkdir(args.odir)
        logging.debug('root: made output directory {0}'.format(args.odir))
    else:
        os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args

def main():
    args = parse_args()

    result = pickle.load(open(args.ifile, 'rb'))

    print(result['aupr'], result['auroc'])

    plt.rc('font', family='Arial', size=11)  # set to Arial/Helvetica
    plt.rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(8, 8), dpi=100)

    ax = fig.add_subplot(1, 1, 1)
    fpr, tpr = result['roc']

    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1])

    ax.set_xlabel('fpr')
    ax.set_ylabel('tpr')

    plt.savefig(os.path.join(args.odir, 'ROC_curve.eps'), dpi = 100)
    plt.close()

    plt.rc('font', family='Arial', size=11)  # set to Arial/Helvetica
    plt.rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(8, 8), dpi=100)

    ax = fig.add_subplot(1, 1, 1)
    precision, recall = result['pr']

    ax.plot(recall, precision)

    ax.set_xlabel('recall')
    ax.set_ylabel('precision')

    plt.savefig(os.path.join(args.odir, 'PR_curve.eps'), dpi = 100)
    plt.close()

    cm = result['confusion matrix']
    cm = np.array(cm, dtype = np.float32)

    cm[0,:] /= np.sum(cm[0,:])
    cm[1, :] /= np.sum(cm[1, :])

    plt.rc('font', family='Arial', size=11)  # set to Arial/Helvetica
    plt.rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(8, 8), dpi=100)

    ax = fig.add_subplot(1, 1, 1)

    display_labels = ['native', 'introgressed']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)

    disp.plot(ax = ax)
    plt.savefig(os.path.join(args.odir, 'confusion_matrix.eps'), dpi = 100)
    plt.close()

if __name__ == '__main__':
    main()
