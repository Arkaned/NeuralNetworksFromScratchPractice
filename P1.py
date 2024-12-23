# in this example, we are mimicking a single neuron node
# it has 3 inputs, a weight for each input, 1 bias, and an output

inputs = [1, 2, 3, 2.5] # each of these unique values could come from different parameters if its the first set of neurons (i.e. heat, colour, water content etc...)
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
print(output)