import h5py
import numpy as np

import matplotlib.pyplot as plt
import random

ifile = h5py.File('archie_64_data.hdf5', 'r')
keys = list(ifile.keys())
random.shuffle(keys)

low_examples = []
high_examples = []

ix = 0

while (len(low_examples) < 3) or (len(high_examples) < 3):
    params = np.array(ifile[keys[ix] + '/params'])
    Y = np.array(ifile[keys[ix] + '/y'])

    for k in range(len(params)):
        if params[k,1] < 0.25:
            low_examples.append(Y[k,:,:,0])
        elif params[k,1] > 0.85:
            high_examples.append(Y[k,:,:,0])

    ix += 1

fig, axes = plt.subplots(nrows = 3, ncols = 2)

print(len(low_examples))

for k in range(3):
    axes[k,0].imshow(low_examples[k])

for k in range(3):
    axes[k,1].imshow(high_examples[k])

plt.show()