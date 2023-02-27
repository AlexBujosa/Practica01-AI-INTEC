from utils.neuron import Neuron
import numpy as np
# Perceptron input size:
input_size = 4
# Step function (returns 0 if y <= 0, or 1 if y > 0):
step_function = lambda y: 0 if y <= 0 else 1
perceptron = Neuron(num_inputs=input_size, activation_function=step_function)
print("Perceptron's random weights = {} , and random bias = {}".format(perceptron.W, perceptron.b))
x = np.random.rand(input_size).reshape(1, input_size)
print("Input vector : {}".format(x))
y = perceptron.forward(x)
print("Perceptron's output value given \`x\` : {}".format(y))
