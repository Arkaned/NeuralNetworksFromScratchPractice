# time to use OOP to make things more scalable

import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],     # X = inputs
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense: # this system also does not require to transpose weights dataset
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#print(np.random.randn(4, 3))
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)