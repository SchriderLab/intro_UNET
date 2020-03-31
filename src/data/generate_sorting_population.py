import sys, os, argparse, logging
import numpy as np
import itertools

import random
import copy

"""
# choose a metrics by which to compute distance
distance_metric_A = ['cityblock', 'cosine', 'dice', 'jaccard', 'kulsinski', 'rogerstanimoto', 'rusellrao', 'sokalmichener']
distance_metric_B = ['cityblock', 'cosine', 'dice', 'jaccard', 'kulsinski', 'rogerstanimoto', 'rusellrao', 'sokalmichener']

# choose whether to invert the metric
inverse_A = ['True', 'False']
inverse_B = ['True', 'False']

sorting_method_A = ['NN_sort', 'seriate', 'dendrogram_sort (ward)', 'dendrogram_sort (single)', 'dendrogram_sort (average)',
                  'dendrogram_sort (complete)', 'None']
sorting_method_B = ['NN_sort', 'seriate', 'dendrogram_sort (ward)', 'dendrogram_sort (single)', 'dendrogram_sort (average)',
                  'dendrogram_sort (complete)', 'None']

perform_matching = ['True', 'False']
matching_metric = ['cityblock', 'cosine', 'dice', 'jaccard', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener']
inverse = ['True', 'False']
matching_direction = ['AB', 'BA']
"""

# choose a metrics by which to compute distance
distance_metric_A = ['cityblock']
distance_metric_B = ['cityblock', 'cosine', 'dice', 'jaccard', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener']

# choose whether to invert the metric
inverse_A = ['False']
inverse_B = ['False']

sorting_method_A = ['None']
sorting_method_B = ['NN_sort', 'seriate', 'dendrogram_sort (ward)', 'dendrogram_sort (single)', 'dendrogram_sort (average)',
                  'dendrogram_sort (complete)']

perform_matching = ['True']
matching_metric = ['cityblock']
inverse = ['True']
matching_direction = ['AB']


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--n_replicates", default = "100")

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

    combinations = list(itertools.product(distance_metric_A, distance_metric_B, inverse_A, inverse_B, sorting_method_A, sorting_method_B, perform_matching, matching_metric, inverse, matching_direction))
    indices = list(range(len(combinations)))

    lines = ['[population_sorting]\n', 'distance_metric_A = {0}\n', 'distance_metric_B = {0}\n', 'inverse_A = {0}\n',
             'inverse_B = {0}\n', 'sorting_method_A = {0}\n', 'sorting_method_B = {0}\n', '\n', '[matching]\n',
             'perform_matching = {0}\n', 'matching_metric = {0}\n', 'inverse = {0}\n', 'matching_direction = {0}\n', '\n']

    for ix in range(int(args.n_replicates)):
        config_ix = random.choice(indices)

        del indices[indices.index(config_ix)]
        config = combinations[config_ix]

        lines_ = copy.copy(lines)
        counter = 0

        for k in range(len(lines_)):
            if '{0}' in lines_[k]:
                lines_[k] = lines_[k].format(config[counter])
                counter += 1

        ofile = open(os.path.join(args.odir, '{0:06d}'.format(ix)), 'w')

        for line in lines_:
            ofile.write(line)

        ofile.close()

if __name__ == '__main__':
    main()



