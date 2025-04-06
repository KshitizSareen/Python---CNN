import idx2numpy
import numpy as np


from Layers.ConvolutedLayer import ConvolutedLayer
from Layers.FullyConnectedLayer import FullyConnectedLayer
from Layers.MaxPoolingLayer import MaxPoolingLayer
from Network import Network
import csv


def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


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

    with open("Error Progression.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Error", "Epoch"])

    learningRate = 0.001

    network = Network()
import idx2numpy
import numpy as np


from Layers.ConvolutedLayer import ConvolutedLayer
from Layers.FullyConnectedLayer import FullyConnectedLayer
from Layers.MaxPoolingLayer import MaxPoolingLayer
from Network import Network
import csv


def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


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

    with open("Error Progression.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Error", "Epoch"])

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


