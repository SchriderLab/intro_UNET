import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Concatenate, BatchNormalization
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D, Conv1D, UpSampling2D
from keras.layers.core import Activation, Reshape, Permute

from keras.layers.pooling import MaxPooling2D, AveragePooling1D, AveragePooling2D, MaxPooling1D
import json

from keras import layers, models, backend
import itertools
import os

from keras import backend as K

import numpy as np
from deeplab_model import *

def relu_clipped(x):
    return K.relu(x, max_value=1)

# function for getting a parabolic list of integers (increasing then decreasing) for the U-Net we're using
def get_filters(k0, alpha):
    ret = [k0]

    for k in range(3):
        ret.append(ret[-1] * alpha)

    for k in range(3):
        ret.append(ret[-1] / alpha)

    return list(map(int, map(np.round, ret)))

def UNET_even_smaller(input_shape = (48, 128, 1), output_activation = relu_clipped):

    inputs = Input(input_shape)
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
    print(model.summary())

    return model

def UNET_deepintraSV(input_shape = (48, 128, 1), batch_normalization = True, k0 = 16, alpha = 2):
    inputs = Input(input_shape)

    filters = get_filters(k0, alpha)

    # ========================
    # convolution 1 (convolve, BN, convolve, BN, pool)
    conv1 = Conv2D(filters[0], 3, activation = 'relu', padding = 'same')(inputs)
    if batch_normalization: conv1 = BatchNormalization()(conv1)

    conv1 = Conv2D(filters[0], 3, activation = 'relu', padding = 'same')(conv1)
    if batch_normalization: conv1 = BatchNormalization()(conv1)

    # pool
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # ===========================
    # convolution 2 (convolve, BN, convolve, BN, pool)
    conv2 = Conv2D(filters[1], 3, activation = 'relu', padding = 'same')(pool1)
    if batch_normalization: conv2 = BatchNormalization()(conv2)
    
    conv2 = Conv2D(filters[1], 3, activation = 'relu', padding = 'same')(conv2)
    if batch_normalization: conv2 = BatchNormalization()(conv2)

    # pool
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # ============================
    # convolution 3 (convolve, BN, convolve, BN, pool)
    conv3 = Conv2D(filters[2], 3, activation = 'relu', padding = 'same')(pool2)
    if batch_normalization: conv3 = BatchNormalization()(conv3)

    conv3 = Conv2D(filters[2], 3, activation = 'relu', padding = 'same')(conv3)
    if batch_normalization: conv3 = BatchNormalization()(conv3)

    drop3 = Dropout(0.5)(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    
    # ============================
    # convolution 4 (convolve, BN, convolve, BN, pool)
    conv4 = Conv2D(filters[3], 3, activation = 'relu', padding = 'same')(pool3)
    if batch_normalization: conv4 = BatchNormalization()(conv4)

    conv4 = Conv2D(filters[3], 3, activation = 'relu', padding = 'same')(conv4)
    if batch_normalization: conv4 = BatchNormalization()(conv4)

    drop4 = Dropout(0.5)(conv4)

    up5 = UpSampling2D(size = (2,2))(drop4)
    merge5 = concatenate([drop3, up5], axis = 3)

    conv5 = Conv2D(filters[4], 3, activation = 'relu', padding = 'same')(merge5)
    if batch_normalization: conv5 = BatchNormalization()(conv5)

    conv5 = Conv2D(filters[4], 3, activation = 'relu', padding = 'same')(conv5)
    if batch_normalization: conv5 = BatchNormalization()(conv5)

    up6 = UpSampling2D(size = (2,2))(conv5)
    merge6 = concatenate([conv2, up6], axis = 3)

    conv6 = Conv2D(filters[5], 3, activation = 'relu', padding = 'same')(merge6)
    if batch_normalization: conv6 = BatchNormalization()(conv6)

    conv6 = Conv2D(filters[5], 3, activation = 'relu', padding = 'same')(conv6)
    if batch_normalization: conv6 = BatchNormalization()(conv6)

    up7 = UpSampling2D(size = (2,2))(conv6)
    merge7= concatenate([conv1, up7], axis = 3)

    conv7 = Conv2D(filters[6], 3, activation = 'relu', padding = 'same')(merge7)
    if batch_normalization: conv7 = BatchNormalization()(conv7)

    conv7 = Conv2D(filters[6], 3, activation = 'relu', padding = 'same')(conv7)
    if batch_normalization: conv7 = BatchNormalization()(conv7)

    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv7)
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

    model = Model(input = inputs, output = conv9)

    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def SegNet(input_shape=(360, 480, 3), classes=1):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder

    # block 1
    ##############################
    x = Conv2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # block 2
    ##############################
    x = Conv2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # block 3
    ##############################
    x = Conv2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # block 4
    ############################
    x = Conv2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # ===========================================================


    # Decoder

    # block 1
    #############################################
    x = Conv2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(classes, 1, 1, border_mode="valid")(x)
    #x = Reshape((input_shape[0] * input_shape[1], classes))(x)
    x = Activation("sigmoid")(x)
    model = Model(img_input, x)
    return model
    
if __name__ == '__main__':

    model = UNET_deepintraSV(input_shape = (64, 128, 1), batch_normalization = True, k0 = 64, alpha = 2)

    print(model.summary())

    jss = model.to_json()

    obj = json.loads(jss)
    jss = json.dumps(obj, indent = 4, sort_keys = True)

    f = open('deepintraSV_k0_64.json', 'w')
    f.write(jss)
    f.close()

    model = SegNet(input_shape = (64, 128, 1))

    print(model.summary())

    jss = model.to_json()

    obj = json.loads(jss)
    jss = json.dumps(obj, indent = 4, sort_keys = True)

    f = open('SegNet_v0.1.json', 'w')
    f.write(jss)
    f.close()



