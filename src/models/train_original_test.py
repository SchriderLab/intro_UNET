import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import preprocessing
from keras import optimizers
from keras import backend as K
from keras.optimizers import SGD
from keras.layers.merge import concatenate
from random import shuffle
from matplotlib import pyplot as plt

data = np.load('training.data.npz')
#s12mat,s21mat,sNomat,s12pos,s21pos,sNopos,s12_target,s21_target,sNo_target = [data[i] for i in ['s12mat', 's21mat', 'sNomat', 's12pos', 's21pos', 'sNopos', 's12_target', 's21_target', 'sNo_target']]

xdata = data['x']
ytarget = data['target']
s = xdata.shape

xdata,ytarget = np.reshape(xdata, newshape=(s[0], s[1], s[2], 1)), np.reshape(ytarget, newshape=(s[0], s[1], s[2], 1))

xdata_train,ytarget_train = xdata[1000:], ytarget[1000:]
xdata_test, ytarget_test  = xdata[:1000], ytarget[:1000] 

xdata_test.shape

def relu_clipped(x):
    return K.relu(x, max_value=1)

inputs = Input((s[1],s[2],1))
conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(inputs)
conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(pool1)
conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(pool2)
conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same')(pool3)
conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same')(pool4)
conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(32, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
merge6 = concatenate([drop4,up6], axis = 3)
conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merge6)
conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv6)

up7 = Conv2D(32, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merge7)
conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv7)

up8 = Conv2D(32, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merge8)
conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv8)

up9 = Conv2D(32, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merge9)
conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv9)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)
conv10 = Conv2D(1, 1, activation = relu_clipped)(conv9)

model = Model(input = inputs, output = conv10)

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(xdata_train, ytarget_train, batch_size=64,
          epochs=25, verbose=1,
          validation_data=(xdata_test, ytarget_test))
