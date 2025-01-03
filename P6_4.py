# one important issue that needs to be accounted for in softmax activation functions is that e**x can lead to very high numbers
# the way to solve this issue of increasingly higher numbers being calculated is:
# overflow prevention: v = u - max(u) so that the max value is 1 and every other value going into e**x is between 0 and 1
# this is done prior to exponentiation

# working on code from P5_2 for this

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense: # this system also does not require to transpose weights dataset
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# define data
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3) # because we only have X Y as inputs, first parameter for LayerDense must be 2. the next can be whatever
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()


#time to run the network...
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5]) # most activations end up being as 0.333... 
# meaning probabilities are starting off evenly distributed...
# to start making corrections we need to learn about the loss function (in P7)