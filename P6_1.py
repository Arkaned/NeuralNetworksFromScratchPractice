# time to learn about the softmax activation function
import math


layer_outputs = [4.8, 1.21, 2.385] # outputs are measured in correctness by the relative contrast between different output neurons outputs.

#layer_outputs = [4.8, 4.79, 4.25]

E = math.e # for creating a softmax activation function of e^x

exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)

print(exp_values)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print(norm_values)
print(sum(norm_values))


