import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import numpy as np
import os

from Layers.FullyConnectedLayer import FullyConnectedLayer
from Layers.ConvolutedLayer import ConvolutedLayer
from Layers.MaxPoolingLayer import MaxPoolingLayer
from Network import Network
from PIL import Image

def readData(path):
    img_matrix = Image.open(path)
    # Define the new size, for example, half the original dimensions
    width, height = img_matrix.size

    resized_img = img_matrix.resize((512,512))
    return np.array(resized_img)

def readTrainingData():
    images = []
    # List of tuples with folder path and corresponding label
    training_folders = [
        ('./Training/glioma_tumor', np.array([1,0,0,0])),
        ('./Training/meningioma_tumor', np.array([0,1,0,0])),
        ('./Training/no_tumor', np.array([0,0,1,0])),
        ('./Training/pituitary_tumor', np.array([0,0,0,1])),
    ]
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    for folder, label in training_folders:
        for filename in os.listdir(folder):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(folder, filename)
                image = readData(image_path)
                images.append([image, label])
    
    return images

images = readTrainingData()


network = Network(0.5)
network.addLayer(ConvolutedLayer(2,2,2,512,512))
network.addLayer(MaxPoolingLayer(2,2))
network.addLayer(ConvolutedLayer(2,2,2,510,510))
network.addLayer(MaxPoolingLayer(2,2))
network.addLayer(FullyConnectedLayer(516128,network.layers[-1],True,False))
network.addLayer(FullyConnectedLayer(2,network.layers[-1],False,False))
network.addLayer(FullyConnectedLayer(2,network.layers[-1],False,False))
network.addLayer(FullyConnectedLayer(4,network.layers[-1],False,True))

# A flat list of 16 values
values = [1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12,
          13, 14, 15,16]

# Create an array and reshape it to 4x4x1
matrix = np.array(values).reshape(4, 4, 1)
network.trainNetwork(1,images)
