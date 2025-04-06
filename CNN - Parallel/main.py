import gzip
import idx2numpy
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import numpy as np
import os

from PIL import Image

from Layers.ConvolutedLayer import ConvolutedLayer
from Layers.FullyConnectedLayer import FullyConnectedLayer
from Layers.MaxPoolingLayer import MaxPoolingLayer
from Network import Network


def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x

def transform_image(image):
    """Convert image from shape (a, b, 3) to (3, a, b)."""
    return np.transpose(image, (2, 0, 1))

def readData(path):
    img_matrix = Image.open(path)
    resized_img = minmax_normalize(img_matrix.resize((28,28)))
    return np.array(resized_img)


def load_mnist():
    X_train = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')

    train_images = []                                                   # reshape train images so that the training set
    for i in range(100):                                # is of shape (60000, 1, 28, 28)
        output=[0 for i in range(10)]
        output[train_labels[i]] = 1
        train_images.append([minmax_normalize(np.expand_dims(X_train[i], axis=2)),np.array(output)])
    return train_images


if __name__ == '__main__':

    images = load_mnist()

    """
    network = Network()
    network.add_layer(Convolutional(name='conv1', num_filters=32, stride=1, size=3, activation='relu'))
    network.add_layer(Pooling(name='pool1', stride=2, size=2))
    network.add_layer(Convolutional(name='conv3', num_filters=64, stride=1, size=3, activation='relu'))
    network.add_layer(Pooling(name='pool2', stride=2, size=2))
    network.add_layer(FullyConnected(name='fullyconnected', nodes1=1600, nodes2=256, activation='relu'))
    network.add_layer(Dense(name='dense', nodes=256, num_classes=10))
    """

    learningRate = 0.001

    network = Network()
    network.addLayer(ConvolutedLayer(500,3,3,28,28,learningRate,1))
    network.addLayer(MaxPoolingLayer(2,2,2,network.layers[-1]))
    network.addLayer(ConvolutedLayer(500,3,3,13,13,learningRate,1))
    network.addLayer(MaxPoolingLayer(2,2,2,network.layers[-1]))
    network.addLayer(FullyConnectedLayer(1600,network.layers[-1],learningRate,False))
    network.addLayer(FullyConnectedLayer(256,network.layers[-1],learningRate,False))
    network.addLayer(FullyConnectedLayer(10,network.layers[-1],learningRate,True))

    network.trainNetwork(1000000,images)


