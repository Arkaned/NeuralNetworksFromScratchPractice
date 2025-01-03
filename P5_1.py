# time to get into activation functions
# particularly rectified linear and step/save functions


import numpy as np

# step activation function be like: if  x > 0: H = 1
# sigmoid functions are a more gradual function, more reliable and adaptable and higher granularity:
# sigmoid functions can be represented as H = 1/(1+e^-x) for example...
# rectified linear is faster than sigmoid but simpler: if x<0: H=0, else: H=X. Creates following shape: _/

# the reason why we use an activation function is like rectified linear or sigmoid is in order to fit better to functions that are non-linear
# rectified linear function is a short cut that removes granularity from sigmoid function.

X = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    output.append(max(0, i))

print(output) # this illustrates the step activation function

