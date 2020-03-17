import numpy as np
from random import shuffle
import gzip
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist, pdist
import json,pprint

from scipy.optimize import linear_sum_assignment
import random

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
def get_windows_snps(x, ipos):
    ipos = list(ipos)

    indices = range(x.shape[1])
    middle_index = findMiddle(indices)

    indices = range(middle_index - 64, middle_index + 64)
    sets = []

    # kind of janky (but whatever)
    # for a predictor image with width N, there 2N - 1 unique windows of size N that include one of those SNPs
    for ix in indices[:63]:
        sets.append(ipos[ix + 1 - 128:ix + 1])

    for ix in indices:
        sets.append(ipos[ix - 64: ix + 64])

    for ix in indices[64:]:
        sets.append(ipos[ix:ix + 128])

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

def min_diff_indices(amat, method = 'min'):
    mb = NearestNeighbors(len(amat), metric = 'manhattan').fit(amat)
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

def sort_cdist(X, metric='cityblock', opt='min', sort_pop=True):
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

