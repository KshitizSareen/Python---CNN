import numpy as np
from LayersCNN.ConvolutionLayer import ConvolutionLayer
from LayersCNN.FullyConnectedLayer import FullyConnectedLayer
from LayersCNN.MaxPoolLayer import  MaxPoolLayer
from NetworkCNN.NeuralNetwork import NeuralNetwork


class NetworkBuilder:
    def __init__(self, input_rows, input_cols, scale_factor):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.scale_factor = scale_factor
        self.layers = []
    
    def add_convolution_layer(self, num_filters, filter_size, step_size, learning_rate, seed):
        if not self.layers:
            self.layers.append(ConvolutionLayer(filter_size, step_size, 1, self.input_rows, self.input_cols, seed, num_filters, learning_rate))
        else:
            prev = self.layers[-1]
            self.layers.append(ConvolutionLayer(filter_size, step_size, prev.get_output_length(), prev.get_output_rows(), prev.get_output_cols(), seed, num_filters, learning_rate))
    
    def add_max_pool_layer(self, window_size, step_size):
        if not self.layers:
            self.layers.append(MaxPoolLayer(step_size, window_size, 1, self.input_rows, self.input_cols))
        else:
            prev = self.layers[-1]
            self.layers.append(MaxPoolLayer(step_size, window_size, prev.get_output_length(), prev.get_output_rows(), prev.get_output_cols()))
    
    def add_fully_connected_layer(self, out_length, learning_rate, seed):
        if not self.layers:
            self.layers.append(FullyConnectedLayer(self.input_cols * self.input_rows, out_length, seed, learning_rate))
        else:
            prev = self.layers[-1]
            self.layers.append(FullyConnectedLayer(prev.get_output_elements(), out_length, seed, learning_rate))
    
    def build(self):
        return NeuralNetwork(self.layers, self.scale_factor)