import numpy as np

class NeuralNetwork:
    def __init__(self, layers, scale_factor):
        self.layers = layers
        self.scale_factor = scale_factor
        self.link_layers()
    
    def link_layers(self):
        if len(self.layers) <= 1:
            return
        
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].set_next_layer(self.layers[i + 1])
            elif i == len(self.layers) - 1:
                self.layers[i].set_previous_layer(self.layers[i - 1])
            else:
                self.layers[i].set_previous_layer(self.layers[i - 1])
                self.layers[i].set_next_layer(self.layers[i + 1])
    
    def get_errors(self, network_output, correct_answer):
        expected = np.zeros(len(network_output))
        expected[correct_answer] = 1
        return network_output - expected
    
    def get_max_index(self, values):
        return np.argmax(values)
    
    def guess(self, image):
        input_list = [image.get_data() / self.scale_factor]
        out = self.layers[0].get_output(input_list)
        return self.get_max_index(out)
    
    def test(self, images):
        correct = sum(1 for img in images if self.guess(img) == img.get_label())
        return correct / len(images)
    
    def train(self, images):
        for img in images:
            input_list = [img.get_data() / self.scale_factor]
            out = self.layers[0].get_output(input_list)
            dLdO = self.get_errors(out, img.get_label())
            self.layers[-1].back_propagation(dLdO)
