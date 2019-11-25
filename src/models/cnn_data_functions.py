import sys, os
import numpy as np
import random

from scipy.spatial.distance import cdist

#import matplotlib.pyplot as plt
import networkx as nx
import itertools

from keras import backend as K

from scipy.sparse import csr_matrix

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def n_snps(ifile):
    fname = os.path.basename(ifile)

    return int(fname.split('_')[-1].split('.')[0])

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def rm_broken_sims(npzFileName):
    try:
        np.load(npzFileName)
        return False
    except Exception as e:
        print('Uh oh. Exception: {}. Going to try to remove file'.format(e))
        os.remove(npzFileName)
        print('Assume we deleted: {}'.format(npzFileName))
        return True

def get_partition_indices(batches, testProp, valProp):
    ret = dict()

    n_batches = len(batches)
    
    n_train = int(np.floor((1 - (testProp + valProp))*n_batches))
    n_val = int(np.floor(valProp*n_batches))
    n_test = int(np.floor(testProp*n_batches))

    ret['train'] = np.random.choice(batches, n_train, replace = False)
    batches = list(set(batches).difference(ret['train']))

    ret['val'] = np.random.choice(batches, n_val, replace = False)
    batches = list(set(batches).difference(ret['val']))

    ret['test'] = np.random.choice(batches, n_test, replace = False)
    batches = list(set(batches).difference(ret['test']))

    return ret

def get_dist_snps(positions):
    distsBetweenSnpVectors = positions[1:] - positions[:-1]
    return distsBetweenSnpVectors


def pad_matrix(currX, positions, max_snps):
    
    pad_len = max_snps - positions.shape[0]

    positions_np = np.zeros(max_snps)

    positions_np[:positions.shape[0]] = positions
    currX = np.pad(currX, ((0, 0), (0, pad_len)), 'constant', constant_values=0)
    return currX, positions_np


def downsize_locus(currCurrX, currCurrPosX, locus_fraction):

    keep_n_sites = int(locus_fraction * len(currCurrPosX))

    downsized_currPosX = currCurrPosX[0:keep_n_sites]
    downsized_currX = currCurrX[:, 0:keep_n_sites]

    return downsized_currX, downsized_currPosX, keep_n_sites


def resort_min_diff(amat):
    
    mb = NearestNeighbors(len(amat), metric = 'manhattan').fit(amat)
    v = mb.kneighbors(amat)

    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]

# Sorts each population according to the Manhattan or 'Cityblock' distance
# Expects that the classes each have the same number of elements
def sort_nn_pop(X, n_classes = 2, metric = 'manhattan'):
    for ix in range(n_classes):
        _ = X[ix*(len(X) // n_classes) : (ix+1)*(len(X) // n_classes)]

        mb = NearestNeighbors(len(_), metric = metric).fit(_)
        v = mb.kneighbors(_)

        smallest = np.argmin(v[0].sum(axis=1))

        X[ix*(len(X) // n_classes) : (ix+1)*(len(X) // n_classes)] = _[v[1][smallest]]

    return X


# For the case of the three-branched networks, (read-compare-decide networks)
# Takes two matrices and sorts the rows s.t. some distance metric is optimized
"""
def sort_cdist(X, metric = 'cityblock', opt = 'min'):
    X1 = X[: X.shape[0] // 2]
    X2 = X[X.shape[0] // 2 :]
    
    D = cdist(X1, X2, metric)

    ret1 = np.zeros(X1.shape)
    ret2 = np.zeros(X2.shape)

    s1 = []
    s2 = []

    k = 0

    if opt == 'min':
        fun = np.nanargmin
    elif opt == 'max':
        fun = np.nanargmax

    while k < X1.shape[0]:
        i, j = np.unravel_index(fun(D), D.shape)

        if (not i in s1) and (not j in s2):

            s1.append(i)
            s2.append(j)

            ret1[k] = X1[i]
            ret2[k] = X2[j]

            k += 1

        D[i,j] = np.nan
        
    return np.vstack((X1, X2))
"""

def sort_cdist(X, metric = 'cityblock', opt = 'min'):
    X1 = X[: X.shape[0] // 2]
    X2 = X[X.shape[0] // 2 :]
    
    D = cdist(X1, X2, metric)

    if opt == 'min':
        D = (D + 0.0001)**-1

    G = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(csr_matrix(D))
    coms = [sorted(u) for u in list(nx.algorithms.matching.max_weight_matching(G))]

    indices = [u[0] for u in coms] + [u[1] for u in coms]

    X = X[indices]

    return X


def format_label_arrays(preformatted_labels):
    preformatted_labels_np = np.array(preformatted_labels)
        
    unique_classes = np.unique(preformatted_labels_np)
    dict_map = {}
    for i in range(len(unique_classes)):
        dict_map[unique_classes[i]] = i
     
    labels = []
    for i in preformatted_labels_np:
        labels.append(dict_map[i])
    labels_np = np.asarray(labels)
    
    min_label = min(labels_np)
    max_label = max(labels_np)
    num_labels = len(np.unique(labels_np))
    int_diff = max_label - min_label
    assert (num_labels == int_diff + 1), "Category labels should be integers, 0 to number of categories - 1"
    return labels


def calc_num_classes(labels):
    num_classes = len(np.unique(labels))
    return num_classes


def format_genotype_arrays(X, posX, X_dtype = 'int8'):
    X = np.array(X, dtype = X_dtype)
    posX = np.array(posX, dtype='float32')
 
    imgRows, imgCols = X.shape[1:]

    return X, posX, imgRows, imgCols


def transform_data2d(currX, currz, positions, sortRows, max_snps, locus_fraction):
    newCurrX, newCurrPosX = [], []
    assert currX.ndim == 2, "There should be one simulations in the saved numpy array."

    currX = np.array([currX])
    positions = [positions]
    ni, nr, nc = currX.shape
    for i in range(ni):
        currCurrX, currCurrPosX = pad_matrix(currX[i], positions[i], max_snps)

        if sortRows == 'nn_pop':
            currCurrX = sort_nn_pop(currCurrX)
        elif sortRows == 'cdist_man_min':
            currCurrX = sort_cdist(currCurrX)

        downsized_currX, downsized_currPosX, keep_n_sites = downsize_locus(currCurrX, currCurrPosX, locus_fraction)

        newCurrX.append(downsized_currX)
        newCurrPosX.append(downsized_currPosX)
    currX = np.array(newCurrX)
    currX = currX.transpose((0,2,1))
    currPosX = np.array(newCurrPosX)

    return currX, currPosX, currz


def transform_data3d(currX, currz, positions, sortRows, max_snps, locus_fraction, locus):
    newCurrX, newCurrPosX = [], []

    assert currX.ndim == 3, "There should be multiple locus simulations in the saved numpy array."
    assert len(currX) == len(positions)
    ni, nr, nc = currX.shape

    currCurrX, currCurrPosX = pad_matrix(currX[locus], positions[locus], max_snps)
    if sortRows:

        currCurrX = resort_min_diff(currCurrX)

    downsized_currX, downsized_currPosX, keep_n_sites = downsize_locus(currCurrX, currCurrPosX, locus_fraction)

    newCurrX.append(downsized_currX)
    newCurrPosX.append(downsized_currPosX)
    currX = np.array(newCurrX)
    currX = currX.transpose((0,2,1))
    currPosX = np.array(newCurrPosX)
    assert currX.shape == (1, keep_n_sites, nr)
    assert currPosX.shape == (1, keep_n_sites) # should this be keep_n_sites-1?
    return currX, currPosX, currz


def read_npz_data(npzFileName):
    try:
        with np.load(npzFileName) as u:
            currX, currz, positions = [u[i] for i in ('X', 'z', 'p')]
    except:
        print('failed to load {}'.format(npzFileName))
        #sys.exit()
    currz = currz.astype(int)

    if 'constant_2pop' in npzFileName:
        assert currz[0] == 3
    elif 'single_pulse_uni_AB' in npzFileName:
        assert currz[0] == 5
    return currX, currz, positions


def create_data_batch(list_IDs_temp, max_snps, n_classes, locus_fraction, locus=None, locus_replicates=1):
    """
    Creates data for CNN input as a batch from list_IDs.
    This function is the same as in data_on_the_fly_classes.DataGenerator.__data_generation,
    but it is not used as a generator function.
    :param list_IDs_temp: list. batch of file names
    :param max_snps: int. max number SNPs from all simulations
    :param n_classes: int. number of classes
    :return: [X, posX], labels
    """

    assert isinstance(list_IDs_temp, (list,)), 'list_IDs_temp must be a list.'

    preformatted_X = []
    preformatted_posX = []
    preformatted_labels = []
    n = 0
    for i, ID in enumerate(list_IDs_temp):
        sortRows = None
        n += 1
        npz_file_name = ID
        currX, currz, positions = read_npz_data(npz_file_name)

        if currX.ndim == 2:
            assert locus_replicates <= 1, 'There is only one locus in {}, but you asked for {}'.format(npz_file_name, locus_replicates)
            currX, currPosX, currz = transform_data2d(currX, currz, positions, sortRows, max_snps, locus_fraction)
        elif currX.ndim == 3:
            assert currX.shape[0] >= locus_replicates, 'There are only {} loci in {}, but you asked for {}'.format(currX.shape[0], npz_file_name, locus_replicates)
            currX, currPosX, currz = transform_data3d(currX, currz, positions, sortRows, max_snps, locus_fraction, locus)
        else:
            print('dimension of genotype matrix should be 2 or 3.')
            sys.exit()

        assert isinstance(currX, np.ndarray), 'Something wrong, data is not an np array'
        assert currX.ndim == 3, 'Dimensions of the genotype data is not correct'

        preformatted_X.extend(currX)
        preformatted_posX.extend(currPosX)
        preformatted_labels.extend(currz)
        labels_array = format_label_arrays(preformatted_labels)
    if len(preformatted_X) == 0:
        print('Exiting, I have no data.')
        sys.exit()
    assert calc_num_classes(labels_array) == n_classes, \
        "Number of labels calculated from numpy array file does not equal number of classes provided in -m argument"

    labels = to_categorical(labels_array, num_classes=n_classes)

    X, posX, imgRows, imgCols = format_genotype_arrays(preformatted_X, preformatted_posX)

    return X, posX, labels


def get_max_snps(inDir):
    """
    Get the maximum number of snps out of all the simulations
    :param inDir:
    :return: max_snps: int of maximum number of snps from all the files
    """
    print('Find max number of snps from simulations')
    max_snps = 0
    for (dirpath, dirnames, filenames) in os.walk(inDir):
        for file_name in filenames:
            if file_name.endswith("maxsnps.txt"):
                file_path = os.path.join(dirpath, file_name)
                with open(file_path) as file:
                    snps = file.readline()
                    if snps:
                        if int(snps) > max_snps:
                            max_snps = int(snps)
                    else:
                        print('WARNING: {} is empty'.format(file_path))
    if max_snps == 0:
        max_snps = None
    return max_snps


def get_ids(inDir, models):
    """
    Get simulation ids from file names
    Warning! This assumes that the first string before "_" defines which model the simulation came from.
    :param inDir:
    :return: list_ids: list of simulation ids
    """
    labels_filename = {}

    infiles = os.path.join(inDir, 'simfiles.txt')
    
    print('Reading simulation files from {}'.format(infiles))
    try:
        f = open(infiles, 'r')
    except FileNotFoundError:
        print('{} does not exist. Please run src/data/create_list_simfiles.py'.format(infiles))
        sys.exit()
    for line in f:
        id = line.rstrip('\n')
        filename = os.path.basename(id)
        path_parts = id.split('/')
        # only include the models indicated in the input argument.
        model = set(models) & set(path_parts)
        if set(models) & set(path_parts):
            labels_filename[id] = list(model)[0]
    f.close()
    assert len(labels_filename) > 0, 'No files from {} added to labels_filename dictionary. Is models -m correct?'.format(infiles)
    return labels_filename


def partition_data(labels_filename, testProp, valProp, num_sims, num_classes):
    """
    Create a dictionary of the simulation files to use for training, validation, and testing.
    :param labels_filename: dict. simulation file names and associated model
    :param testProp: float. Proportion of simulations to use for training
    :param valProp: float. Proportion of simulations to use for validation
    :param num_sims: None or int
    :param num_classes: int. The number of simulated models
    :return: partition: dict. keys are training, validation, and testing.
    Values are lists of lists of simulation file names for each model.
    """

    print('Partitioning the simulation data into training, validation, and testing sets.')
    assert isinstance(labels_filename, dict), 'labels_filename must be dictionary'
    if num_sims is None:
        num_sims = len(labels_filename)

    testSize = int(testProp * num_sims)
    valSize = int(valProp * num_sims)
    n_test = int(testSize / num_classes)
    n_val = int(valSize / num_classes)
    n_train = int((num_sims - testSize - valSize) / num_classes)

    models = set(labels_filename.values())
    assert len(models) == num_classes, \
        "number of classes given in --models does not match number of labels from filename ids. " \
        "Do your file names have the format 'label_id'?"
    partition = {}
    partition_validation = []
    partition_test = []
    partition_train = []
    for model in models:
        total_dict = {key: value for key, value in labels_filename.items() if value == model}
        train_keys = random.sample(list(total_dict), n_train)
        partition_train.append(train_keys)

        validation_set = set(train_keys) ^ set(total_dict.keys())
        validation_keys = random.sample(list(validation_set), n_val)
        partition_validation.append(validation_keys)

        test_set = set(validation_set) ^ set(validation_keys)
        test_keys = random.sample(list(test_set), n_test)
        partition_test.extend(test_keys)

        assert len(set(train_keys) & set(validation_keys)) is 0, 'files in training and validation overlap'
        assert len(set(train_keys) & set(test_keys)) is 0, 'files in training and testing overlap'
        assert len(set(test_keys) & set(validation_keys)) is 0, 'files in testing and validation overlap'

    assert len(partition_train) == len(models)
    assert len(partition_validation) == len(models)

    print('{} simulations for training'.format(sum(map(len, partition_train))))
    print('{} simulations for validation'.format(sum(map(len, partition_validation))))
    print('{} simulations for testing'.format(len(partition_test)))

    partition['train'] = partition_train
    partition['validation'] = partition_validation
    partition['test'] = partition_test
    return partition
