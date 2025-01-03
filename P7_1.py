import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import math

# loss function of choice for classification is categorical cross-entropy:
# the function for this is described as the negative sum of the target value multiplied by the log of the predicted value for each of the values in the distribution
# simplifies to be the -log of predicted values of target classes, is simplified using 1 hot coding which will be explained later

# one-hot encoding contains a vector of N classes long, the vector of classes will have 0's for every class except the label/selection.
"""
# ln(x) will be important for this so let's discuss that
solve for x:
e**x = b

"""
b = 5.2
#print(np.log(b))
#print(math.e ** 1.6486586255873816)

# so, the predicted target class (the selected/labeled class in onehot coding removes all other predictions by multiplying by 0)

softmax_output = [0.7, 0.1, 0.2 ] # this is meant to replicate the output from a softmax function

# imagine target class is 0

target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] + 
         math.log(softmax_output[0])*target_output[1] +
         math.log(softmax_output[0])*target_output[2])

print(loss)
loss = -math.log(softmax_output[0])
print(loss) # same thing lol

print(-math.log(0.7))
print(-math.log(0.5))

