import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import keras.backend as K
from keras.layers import Conv2D, ReLU, BatchNormalization, Input, Add, MaxPool2D, UpSampling2D
from keras.models import Sequential, Model
from keras.utils import plot_model
import scipy.io
import numpy as np
from matplotlib import pyplot
import cv2
from os import path
from keras.utils import Sequence
from keras.preprocessing import image


nModules = 2
nFeats = 256
nStack = 4
nOutChannels = 16


# def convBlock(numIn, numOut):
#     model = Sequential()
#     model.add(BatchNormalization())
#     model.add(ReLU())
#     model.add(Conv2D(int(numOut/2), 1, strides=1, padding='same'))
#     model.add(BatchNormalization())
#     model.add(ReLU())
#     model.add(Conv2D(int(numOut/2), 3, strides=1, padding='same'))
#     model.add(BatchNormalization())
#     model.add(ReLU())
#     model.add(Conv2D(numOut, 1, strides=1, padding='same'))
#     return model

def convBlock(inputs, numIn, numOut):
    x = BatchNormalization()(inputs)
    x = ReLU()(x)
    x = Conv2D(int(numOut / 2), 1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(int(numOut / 2), 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(numOut, 1, strides=1, padding='same')(x)
    return x


def skipLayer(inputs, numIn, numOut):
    if numIn == numOut:
        return inputs
    else:
        return Conv2D(numOut, 1, strides=1, padding='same')(inputs)


def Residual(inputs, numIn, numOut):
    add1 = convBlock(inputs, numIn, numOut)
    add2 = skipLayer(inputs, numIn, numOut)
    x = Add()([add1, add2])
    return x


def hourglass(inputs, n, f):
    up1 = inputs
    for i in range(nModules):
        up1 = Residual(up1, f, f)
    low1 = MaxPool2D(2)(inputs)
    for i in range(nModules):
        low1 = Residual(low1, f, f)
    if n > 1:
        low2 = hourglass(low1, n - 1, f)
    else:
        low2 = low1
        for i in range(nModules):
            low2 = Residual(low2, f, f)
    low3 = low2
    for i in range(nModules):
        low3 = Residual(low3, f, f)
    up2 = UpSampling2D(2)(low3)
    x = Add()([up1, up2])
    return x


def lin(inputs, numIn, numOut):
    x = Conv2D(filters=numOut, kernel_size=1, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    return x


def creatModel():
    inputs = Input(shape=[256, 256, 3])
    x = Conv2D(filters=64, kernel_size=7, strides=2, padding='same')(inputs)
    x = ReLU()(x)
    x = Residual(x, 64, 128)
    x = MaxPool2D(2)(x)
    x = Residual(x, 128, 128)
    x = Residual(x, 128, nFeats)

    out = []
    inter = x
    for i in range(nStack):
        hg = hourglass(inter, 4, nFeats)
        ll = hg
        for j in range(nModules):
            ll = Residual(ll, nFeats, nFeats)
        ll = lin(ll, nFeats, nFeats)
        tmpOut = Conv2D(filters=nOutChannels, kernel_size=1, strides=1, padding='same')(ll)
        out.insert(i, tmpOut)

        if i < nStack:
            ll_ = Conv2D(filters=nFeats, kernel_size=1, strides=1, padding='same')(ll)
            tmpOut_ = Conv2D(filters=nFeats, kernel_size=1, strides=1, padding='same')(tmpOut)
            inter = Add()([ll_, tmpOut_])
    model = Model(inputs=inputs, outputs=out)
    return model


def creatModelD():
    inputs = Input([64, 64, nOutChannels + 3])
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(inputs)
    x = ReLU()(x)
    x = Residual(x, 64, 128)
    x = Residual(x, 128, 128)
    x = Residual(x, 128, nFeats)

    out = []

    x = hourglass(x, 4, nFeats)

    ll = x
    for j in range(nModules):
        ll = Residual(ll, nFeats, nFeats)
    ll = lin(ll, nFeats, nFeats)
    tmpOut = Conv2D(filters=nOutChannels, kernel_size=1, strides=1, padding='same')(ll)
    out.insert(0, tmpOut)
    model = Model(inputs=inputs, outputs=out)

    return model


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, filepath, matdir, batch_size=16, dim=(256, 256), n_channels=3,
                 n_joints=14, shuffle=True):
        'Initialization'
        self.mat = scipy.io.loadmat(matdir)
        self.dim = dim
        self.batch_size = batch_size
        self.filepath = filepath
        self.list = os.listdir(filepath)
        self.n_channels = n_channels
        self.n_joints = n_joints
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_joints))
        # y = np.empty((self.batch_size, *self.dim, self.n_classes))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img = image.load_img(self.filepath + ID)
            img = image.img_to_array(img)
            h, w, d = img.shape
            img = cv2.resize(img, (256, 256))
            X[i,] = img

            # Store class
            number = int(ID[2:6])
            Gauss_map = np.zeros([256, 256, 14])
            for k in range(14):
                xs = self.mat['joints'][0][k][number] / w * 256
                ys = self.mat['joints'][1][k][number] / h * 256
                sigma = 1
                mask_x = np.matlib.repmat(xs, 256, 256)
                mask_y = np.matlib.repmat(ys, 256, 256)

                x1 = np.arange(256)
                x_map = np.matlib.repmat(x1, 256, 1)

                y1 = np.arange(256)
                y_map = np.matlib.repmat(y1, 256, 1)
                y_map = np.transpose(y_map)

                temp = ((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2) / (2 * sigma ** 2)

                Gauss_map[:, :, k] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-temp)

            y[i,] = Gauss_map

        return X, y


def main():
    # rootdir = '/data/lsp_dataset/images/'
    # jointsdir = '/data/lsp_dataset/joints.mat'
    # test = DataGenerator('/data/lsp_dataset/images/', '/data/lsp_dataset/joints.mat')

    # input_real = []
    x = Input([256, 256, 256])
    out = hourglass(x, 4, 256)
    model = Model(inputs=x, outputs=out)
    # netD = creatModelD()
    # model.fit_generator()
    model.summary()
    plot_model(model, to_file='model.png')
    # output_real = netD(input_real)

    print('huhu')


if __name__ == '__main__':
    main()
