import numpy as np
from random import shuffle
import gzip
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist, pdist
import json,pprint

from scipy.optimize import linear_sum_assignment

def add_channel(matrix):
    return matrix.reshape(matrix.shape + (1, ))

def min_diff_indices(amat, method = 'min'):
    mb = NearestNeighbors(len(amat), metric = 'manhattan').fit(amat)
    v = mb.kneighbors(amat)

    if method == 'min':
        smallest = np.argmin(v[0].sum(axis=1))
    elif method == 'max':
        smallest = np.argmax(v[0].sum(axis=1))

    return v[1][smallest]

def sort_NN(X):
    X1 = X[: X.shape[0] // 2]
    X2 = X[X.shape[0] // 2:]

    indices = []
    indices.extend(min_diff_indices(X1))
    indices.extend([u + X.shape[0] // 2 for u in min_diff_indices(X2)])

    return X[indices], indices

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
            #print(list(i))
            i = [int(j) for j in i[0]]
            if len(i) > max_len:
                i = i[:max_len]
            else:
                print('aah.  too short at ', gdx)
                break
            #print(len(p))
            #missing = max_len - len(i)
            #for z in range(missing):
            #    i.append(0)
            #    if kdx:
            #        p.append(-1)
            #kdx = 0
            i = np.array(i, dtype=np.int8)
            q.append(i)
            #print(len(p))
        q = np.array(q)
        #f1,f2 = sort_min_diff(q[0:int(nindv/2)]), sort_min_diff(q[int(nindv/2):nindv])
        #q = np.append(f1[0],f2[0], axis=0)
        #row_ord = {jdx:int(uu) for jdx,uu in enumerate(np.append(f1[1], f2[1]+20))}
        #breakD = {}
        #for i in igD[gdx]:
        #    breakD[i] = igD[gdx][row_ord[i]]
        #igD[gdx] = breakD
        #pprint.pprint(breakD)
        #pprint.pprint(igD[gdx])
        #print('*******************************')
        q = q.astype("int8")
        f.append(np.array(q))
        pos_int = np.array(p, dtype='float')
        #print(pos_int)
        #pos.append(pos_int* loc_len**-1)
        mask_mat = []
        breakD = igD[gdx]
        for indv in range(len(breakD)):
            mask = binary_digitizer(pos_int, breakD[indv])
            #print(mask)
            mask_mat.append(mask[:max_len])
            #if breakD[indv]:
            #    print(indv, breakD[indv])
            #    pprint.pprint(list(zip(pos_int,mask)))
            #    print('*'*30)
        #plt.imshow(mask_mat, aspect=14, cmap='bone')
        #plt.show()
        #print(np.array(mask_mat).shape, q.shape)
        target.append(np.array(mask_mat, dtype='int8'))
        if not len(f) % 100: print(len(f), f[-1].shape, target[-1].shape)
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

