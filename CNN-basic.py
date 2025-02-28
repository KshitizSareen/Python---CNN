import gzip
import idx2numpy
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import numpy as np
import os

from layer import Convolutional,Pooling,FullyConnected,Dense
from model import Network
from PIL import Image


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


    
    
    
    return transform_image(np.array(resized_img))

def readTrainingData():
    images = []
    # List of tuples with folder path and corresponding label
    training_folders = [
        ('./Training/glioma_tumor', 0),
        ('./Training/meningioma_tumor', 1),
        ('./Training/no_tumor', 2),
        ('./Training/pituitary_tumor', 3),
    ]
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    for folder, label in training_folders:
        k=0
        for filename in os.listdir(folder):
            if k>50:
                break
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(folder, filename)
                image = readData(image_path)
                images.append([image, label])
            k+=1
    
    return images

def load_mnist():
    X_train = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')

    train_images = []                                                   # reshape train images so that the training set
    for i in range(100):                                # is of shape (60000, 1, 28, 28)
        train_images.append([minmax_normalize(np.expand_dims(X_train[i], axis=0)),train_labels[i]])
    return train_images


images = readTrainingData()






network = Network()
network.add_layer(Convolutional(name='conv1', num_filters=32, stride=1, size=3, activation='relu'))
network.add_layer(Pooling(name='pool1', stride=2, size=2))
network.add_layer(Convolutional(name='conv3', num_filters=64, stride=1, size=3, activation='relu'))
network.add_layer(Pooling(name='pool2', stride=2, size=2))
network.add_layer(FullyConnected(name='fullyconnected', nodes1=1600, nodes2=256, activation='relu'))
network.add_layer(Dense(name='dense', nodes=256, num_classes=10))



# A flat list of 16 values
values = [1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12,
          13, 14, 15,16,
          17,18,19,20,
          21,22,23,24,25]

# Create an array and reshape it to 4x4x1
matrix = np.array(values).reshape(5, 5, 1)
input = [[matrix,np.array([1])] for i in range(10)] 


network.train(images,1000,0.001,False,0,True,10)
