# time to integrate numpy to facilitate this process
# if we want to multiply the list of lists with the inputs, we can use the dot product
# the dot does the same job as P2 and P3_1 multiplying input indexes with each weight index
import numpy as np

inputs = [1, 2, 3, 2.5] 

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = 2.0, 3.0, 0.5

output = np.dot(weights, inputs) + biases
# since weights has multiple arrays within the list, 
# it is importan to put weights first in the np.dot function

print(output)