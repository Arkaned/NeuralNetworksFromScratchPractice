
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# let's create the spiral data with the features and classes
X, y = spiral_data(100, 3) # 100 feature sets of 3 classes

class Layer_Dense: # this system also does not require to transpose weights dataset
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#print(np.random.randn(4, 3))
layer1 = Layer_Dense(2,5) # 2 = number of inputs, 5 = number of neurons
activation1 = Activation_ReLU()

layer1.forward(X)
print(layer1.output)
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)