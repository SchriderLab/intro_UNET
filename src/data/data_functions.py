import numpy as np
from random import shuffle
import gzip
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from scipy.spatial.distance import cdist, pdist, squareform, dice
import json,pprint

from scipy.optimize import linear_sum_assignment
import random

from sklearn import datasets
from fastcluster import linkage

import sys

from seriate import seriate


def seriation(Z, N, cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def compute_serial_matrix(dist_mat, method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


def sort_XY(X, Y, config, return_indices = False):
    x1 = X[:X.shape[0] // 2, :]
    x2 = X[X.shape[0] // 2:, :]

    y1 = Y[:X.shape[0] // 2, :]
    y2 = Y[X.shape[0] // 2:, :]

    D_x1 = pdist(x1, metric = config.get('population_sorting', 'distance_metric_A'))
    D_x2 = pdist(x2, metric = config.get('population_sorting', 'distance_metric_B'))

    D_x1[np.where(np.isnan(D_x1))] = 0.
    D_x2[np.where(np.isnan(D_x2))] = 0.

    if config.getboolean('population_sorting', 'inverse_A'):
        D_x1 = (D_x1 + 0.000001)**-1

    if config.getboolean('population_sorting', 'inverse_B'):
        D_x2 = (D_x2 + 0.000001)**-1

    sorting_method_A = config.get('population_sorting', 'sorting_method_A')
    sorting_method_B = config.get('population_sorting', 'sorting_method_B')

    # sort population A
    ####################################

    if sorting_method_A == 'seriate':
        i1 = seriate(D_x1)

    elif sorting_method_A == 'NN_sort':
        if config.getboolean('population_sorting', 'inverse_A'):
            i1 = min_diff_indices(x1, method = 'max', metric = config.get('population_sorting', 'distance_metric_A'))
        else:
            i1 = min_diff_indices(x1, method='min', metric=config.get('population_sorting', 'distance_metric_A'))

    elif 'dendrogram_sort' in sorting_method_A:
        ordered_distance_mat, i1, _ = compute_serial_matrix(squareform(D_x1), sorting_method_A.split(' ')[-1].replace('(', '').replace(')', ''))

    elif sorting_method_A == 'None':
        i1 = list(range(x1.shape[0]))

    #####################################
    #####################################
    # sort population B

    if sorting_method_B == 'seriate':
        i2 = seriate(D_x2)

    elif sorting_method_B == 'NN_sort':
        if config.getboolean('population_sorting', 'inverse_A'):
            i2 = min_diff_indices(x1, method = 'max', metric = config.get('population_sorting', 'distance_metric_B'))
        else:
            i2 = min_diff_indices(x1, method='min', metric=config.get('population_sorting', 'distance_metric_B'))

    elif 'dendrogram_sort' in sorting_method_B:
        
        ordered_distance_mat, i2, _ = compute_serial_matrix(squareform(D_x2), sorting_method_B.split(' ')[-1].replace('(', '').replace(')', ''))

    elif sorting_method_B == 'None':
        i2 = list(range(x2.shape[0]))

    # do matching if necessary
    #######################################

    x1 = x1[i1]
    x2 = x2[i2]

    y1 = y1[i1]
    y2 = y2[i2]

    if config.getboolean('matching', 'perform_matching'):
        if config.get('matching', 'matching_direction') == 'AB':
            D = cdist(x2, x1, metric = config.get('matching', 'matching_metric'))
            D[np.where(np.isnan(D))] = 0.

            if config.getboolean('matching', 'inverse'):
                D = (D + 0.000001)**-1

            i, j = linear_sum_assignment(D)

            coms = dict(zip(i, j))

            i1_ = [coms[u] for u in sorted(coms.keys())]
            x1 = x1[i1_]
            y1 = y1[i1_]

            # sort i1 in case we need to return it
            i1 = [i1[u] for u in i1_]

        elif config.get('matching', 'matching_direction') == 'BA':
            D = cdist(x1, x2, metric=config.get('matching', 'matching_metric'))
            D[np.where(np.isnan(D))] = 0.

            if config.getboolean('matching', 'inverse'):
                D = (D + 0.000001) ** -1

            i, j = linear_sum_assignment(D)

            coms = dict(zip(i, j))

            i2_ = [coms[u] for u in sorted(coms.keys())]
            x2 = x2[i2_]
            y2 = y2[i2_]

            i2 = [i2[u] for u in i2_]

    if return_indices:
        return np.vstack([x1, x2]), np.vstack([y1, y2]), i1, i2
    else:
        return np.vstack([x1, x2]), np.vstack([y1, y2])

# finds the middle component of a list (if there are an even number of entries, then returns the first of the middle two)
def findMiddle(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        return (input_list[int(middle)], input_list[int(middle-1)])[0]

# something from the ArchIE repository
# a supporting function, writes positions to a file in a tab separated format
def write_snp_file(positions, ofile):
    ofile = open(ofile, 'w')

    line = '1:{0}\t1\t 0.000000\t{0}\tA\tG\n'

    for p in positions:
        ofile.write(line.format(p))

    ofile.close()

# writes a binary matrix to a text file
def write_binary_matrix(mat, ofile):
    ofile = open(ofile, 'w')

    mat = mat.T

    for m in mat:
        ofile.write(''.join(list(map(str,m))) + '\n')

    ofile.close()

# Gets all possible windows over a predictor image with a particular window size
def get_windows(x, ipos, wsize = 500):
    # positions of polymorphisms
    ipos = list(ipos)

    indices = range(x.shape[1])
    # get the middle index
    middle_index = findMiddle(indices)

    # get the indices for the x-axis of a predictor image of 128 SNPs
    indices = range(middle_index - 64, middle_index + 64)
    sets = []

    # for these indices, get as many unique windows (of the specified size) as possible that include these positions
    for ix in indices:
        p = set([u for u in ipos if (u >= ipos[ix] - wsize) and (u <= ipos[ix])])

        if p not in sets:
            sets.append(p)

        p = set([u for u in ipos if (u >= ipos[ix]) and (u <= ipos[ix] + wsize)])

        if p not in sets:
            sets.append(p)

    # sort
    sets = [sorted(list(u)) for u in sets]
    sets = sorted(sets, key = lambda u: u[0])

    ret = []

    for s in sets:
        ret.append([ipos.index(u) for u in s])

    return ret, indices

# same idea as above function, but here the window size is fixed in terms of SNPs (128 SNPs) and
# not base pairs
def get_windows_snps(x, ipos, N = 128):
    ipos = list(ipos)

    indices = range(x.shape[1])
    middle_index = findMiddle(indices)

    indices = range(middle_index - N // 2, middle_index + N // 2)
    sets = []

    # kind of janky (but whatever)
    # for a predictor image with width N, there 2N - 1 unique windows of size N that include at least one of those SNPs
    for ix in indices[:(N // 2 - 1)]:
        sets.append(ipos[ix + 1 - N:ix + 1])

    for ix in indices:
        sets.append(ipos[ix - N // 2: ix + N // 2])

    for ix in indices[(N // 2):]:
        sets.append(ipos[ix:ix + N])

    sets = [sorted(list(u)) for u in sets]
    sets = sorted(sets, key = lambda u: u[0])

    ret = []

    for s in sets:
        ret.append([ipos.index(u) for u in s])

    return ret, indices

# reads params from an SLiM output file
# this one is for the two-pop introgressive hybridization problem problem where there are 3 parameters per replicate:
# Migration time, migration probability (A -> B), and migration probability (B -> A)
def get_params(out):
    ret = []

    # migTime, migProb12, migProb21
    _ = []

    ifile = open(out, 'r')
    line = ifile.readline()

    while line != '':
        if 'migTime' in line:
            _.append(int(line.replace('migTime: ', '').replace('\n', '')))
        elif 'migProbs' in line:
            _.extend(list(map(float, line.replace('migProbs:', '').replace('\n', '').split(','))))

            ret.append(_)
            _ = []

        line = ifile.readline()

    return np.array(ret)

# for the ArchIE problem
# read parameters from a SLiM output file (the stdout from a simulation)
# two parameters to be read: migration time, and migration probability (from the ghost population)
def get_params_ghost(out):
    ret = []

    # migTime, migProb12, migProb21
    _ = []

    ifile = open(out, 'r')
    line = ifile.readline()

    while line != '':
        if 'migTime' in line:
            _.append(int(line.replace('migTime: ', '').replace('\n', '')))
        elif 'migProb' in line:
            _.append(float(line.replace('migProb: ', '').replace('\n', '')))

            ret.append(_)
            _ = []

        line = ifile.readline()

    return np.array(ret)

# add a channel axis to a 2D matrix
def add_channel(matrix):
    return matrix.reshape(matrix.shape + (1, ))

# compare to different predictor and target images
# was used to visually compare different sortings
def plot_sorting(x, y, xo, yo):
    fig, axes = plt.subplots(nrows = 2, ncols = 2)

    axes[0][0].imshow(x, cmap = 'gray')
    axes[0][1].imshow(y, cmap = 'gray')

    axes[1][0].imshow(xo, cmap = 'gray')
    axes[1][1].imshow(yo, cmap = 'gray')

    plt.show()
    plt.close()

def cosine_distance(x, y):
    _ = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    if _ == np.nan:
        return 0.
    else:
        return _

def dice_distance(x, y):
    _ = dice(x, y)

    if _ == np.nan:
        return 0.
    else:
        return _

def min_diff_indices(amat, method = 'min', metric = 'cityblock'):
    if metric == 'cityblock':
        metric = 'manhattan'
    elif metric == 'cosine':
        metric = cosine_distance
    elif metric == 'dice':
        metric = dice_distance

    mb = NearestNeighbors(len(amat), metric = metric).fit(amat)
    v = mb.kneighbors(amat)

    if method == 'min':
        smallest = np.argmin(v[0].sum(axis=1))
    elif method == 'max':
        smallest = np.argmax(v[0].sum(axis=1))

    return v[1][smallest]

def sort_NN(X, method = 'min'):
    X1 = X[: X.shape[0] // 2]
    X2 = X[X.shape[0] // 2:]

    indices = []
    indices.extend(min_diff_indices(X1, method = method))
    indices.extend([u + X.shape[0] // 2 for u in min_diff_indices(X2, method = method)])

    return X[indices], indices

def shuffle_indices(X):
    i1 = list(range(X.shape[0] // 2))
    random.shuffle(i1)

    i2 = list(range(X.shape[0] // 2, X.shape[0]))
    random.shuffle(i2)

    return i1 + i2

def sort_cdist(X, metric='cityblock', opt='min', sort_pop=True, reverse_pops = False):
    X1 = X[: X.shape[0] // 2]
    X2 = X[X.shape[0] // 2:]

    D = cdist(X1, X2, metric)

    if opt == 'max':
        D = (D + 0.0001) ** -1

    i, j = linear_sum_assignment(D)

    coms = dict(zip(i, j))

    if sort_pop:
        pop1_indices = min_diff_indices(X1)
    else:
        pop1_indices = sorted(coms.keys())

    pop2_indices = [coms[u] + (X.shape[0] // 2) for u in pop1_indices]

    indices = list(pop1_indices) + list(pop2_indices)

    X = X[indices]

    return X, indices

def sort_min_diff(amat):
    mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]], v[1][smallest]

def get_gz_file(filename, splitchar = 'NA', buffered = False):
    print(filename)
    if not buffered:
        if splitchar == 'NA':
            return [i.strip().split() for i in gzip.open(filename, 'rt')]
        else: return [i.strip().split(splitchar) for i in gzip.open(filename, 'rt')]
    else:
        if splitchar == 'NA':
            return (i.strip().split() for i in gzip.open(filename, 'rt'))
        else: return (i.strip().split(splitchar) for i in gzip.open(filename, 'rt'))

def binary_digitizer(x, breaks):
    #x is all pos of seg sites
    #breaks are the introgression breakpoints, as a list of lists like [[1,4], [22,57], [121,144]....]
    #output is a numpy vector with zero for all pos not in introgression and one for all points in introgression
    flat_breaks = np.array(breaks).flatten()
    lenx = len(x)
    lzero, rzero = np.zeros(lenx), np.zeros(lenx)
    dg_l = np.digitize(x, flat_breaks, right=False)
    dg_r = np.digitize(x, flat_breaks, right=True)
    lzero[dg_l % 2 > 0] = 1
    rzero[dg_r % 2 > 0] = 1
    return np.array([lzero, rzero]).max(axis=0)

def load_data(msfile, introgressfile, max_len, nindv):
    ig = list(get_gz_file(introgressfile))
    igD = {}
    for x in ig:
        if x[0] == 'Begin':
            n = int(x[-1])
            igD[n] = {}
        if x[0] == 'genome':
            if len(x) > 2:
                igD[n][int(x[1].replace(":", ""))] = [tuple(map(int,i.split('-'))) for i in x[-1].split(',')]
            else:  igD[n][int(x[1].replace(":", ""))] = []           #print(n,x)
    #pprint.pprint(igD)
    g = list(get_gz_file(msfile))
    loc_len = 10000.
    #print(loc_len)
    k = [idx for idx,i in enumerate(g) if len(i) > 0 and i[0].startswith('//')]
    #print(k)
    f, pos, target = [], [], []
    for gdx,i in enumerate(k):
        L = g[i+3:i+3+nindv]
        p = [jj for jj in g[i+2][1:]][:max_len]
        q = []
        kdx = 1
        for i in L:
            i = [int(j) for j in i[0]]
            if len(i) > max_len:
                i = i[:max_len]
            else:
                raise Exception("Sorry too short")
                print('aah.  too short at ', gdx)
                break

            i = np.array(i, dtype=np.int8)
            q.append(i)

        q = np.array(q)

        q = q.astype("int8")
        f.append(np.array(q))
        pos_int = np.array(p, dtype='float')

        mask_mat = []
        breakD = igD[gdx]
        for indv in range(len(breakD)):
            mask = binary_digitizer(pos_int, breakD[indv])
            mask_mat.append(mask[:max_len])

        target.append(np.array(mask_mat, dtype='int8'))
    return f, pos, target, igD

def load_data_ghost(msfile, introgressfile, max_len, nindv):
    ig = list(get_gz_file(introgressfile))
    igD = {}
    for x in ig:
        if x[0] == 'Begin':
            n = int(x[-1])
            igD[n] = {}
        if x[0] == 'genome':
            if len(x) > 2:
                igD[n][int(x[1].replace(":", ""))] = [tuple(map(int,i.split('-'))) for i in x[-1].split(',')]
            else:  igD[n][int(x[1].replace(":", ""))] = []           #print(n,x)
    #pprint.pprint(igD)
    g = list(get_gz_file(msfile))
    loc_len = 10000.
    #print(loc_len)
    k = [idx for idx,i in enumerate(g) if len(i) > 0 and i[0].startswith('//')]
    #print(k)
    f, pos, target = [], [], []
    for gdx,i in enumerate(k):
        L = g[i+3:i+3+nindv]
        p = [jj for jj in g[i+2][1:]]
        q = []
        kdx = 1
        for i in L:
            i = [int(j) for j in i[0]]

            i = np.array(i, dtype=np.int8)
            q.append(i)

        q = np.array(q)

        q = q.astype("int8")
        f.append(np.array(q))
        pos_int = np.array(p, dtype='float')

        pos.append(pos_int)

        mask_mat = []
        breakD = igD[gdx]
        for indv in range(len(breakD)):
            mask = binary_digitizer(pos_int, breakD[indv])
            mask_mat.append(mask)

        target.append(np.array(mask_mat, dtype='int8'))
    return f, pos, target, igD

def split(word):
    return [char for char in word]

def up_sample_populations(pop1_x, pop2_x, pop1_y, pop2_y, up_sample_pop_size = 32):
    pop1_indices = list(range(pop1_x.shape[0])) + list(np.random.choice(range(pop1_x.shape[0]), size = up_sample_pop_size - pop1_x.shape[0], replace = False))
    random.shuffle(pop1_indices)

    pop2_indices = list(range(pop2_x.shape[0])) + list(np.random.choice(range(pop2_x.shape[0]), size=up_sample_pop_size - pop2_x.shape[0], replace = True))
    random.shuffle(pop2_indices)

    return pop1_x[pop1_indices], pop2_x[pop2_indices], pop1_y[pop1_indices], pop2_y[pop2_indices]

def load_data_dros(msFile, ancFile, n_sites = 64, up_sample = False, up_sample_pop_size = 32):
    msFile = open(msFile, 'r')
    ancFile = open(ancFile, 'r')

    ms_lines = msFile.readlines()

    idx_list = [idx for idx, value in enumerate(ms_lines) if '//' in value] + [len(ms_lines)]

    ms_chunks = [ms_lines[idx_list[k]:idx_list[k+1]] for k in range(len(idx_list) - 1)]
    ms_chunks[-1] += ['\n']

    anc_lines = ancFile.readlines()

    X = []
    Y = []

    for chunk in ms_chunks:
        pos = np.array([u for u in chunk[2].split(' ')[1:-1] if u != ''], dtype = np.float32)

        x = np.array([list(map(int, split(u.replace('\n', '')))) for u in chunk[3:-1]], dtype = np.uint8)
        y = np.array([list(map(int, split(u.replace('\n', '')))) for u in anc_lines[:len(pos)]], dtype = np.uint8)

        y = y.T

        del anc_lines[:len(pos)]

        if n_sites is not None:
            pop1_x = x[:20, :n_sites]
            pop2_x = x[20:, :n_sites]

            pop1_y = y[:20, :n_sites]
            pop2_y = y[20:, :n_sites]
        else:
            pop1_x = x[:20, :]
            pop2_x = x[20:, :]

            pop1_y = y[:20, :]
            pop2_y = y[20:, :]

        if up_sample:
            pop1_x, pop2_x, pop1_y, pop2_y = up_sample_populations(pop1_x, pop2_x, pop1_y, pop2_y, up_sample_pop_size = up_sample_pop_size)

            x = np.vstack((pop1_x, pop2_x))
            y = np.vstack((pop1_y, pop2_y))

        X.append(x)
        Y.append(y)

    return X, Y

def max_len_only(xfile):
    g = list(get_gz_file(xfile))
    k = [idx for idx,i in enumerate(g) if len(i) > 0 and str(i[0]).startswith('//')]
    #print(k)
    ml = 0
    for i in k:
        L = g[i+3:i+37]
        q = []
        for i in L:
            if len(i[0]) > ml: ml = len(i[0])
    return ml

