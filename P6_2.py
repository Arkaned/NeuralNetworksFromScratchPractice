# now convert the previous work into numpy

import math 
import numpy as np

layer_outputs = [4.8, 1.21, 2.385] # outputs are measured in correctness by the relative contrast between different output neurons outputs.

E = math.e # for creating a softmax activation function of e^x

exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values))