from utils.neuron import Neuron, sigmoid_function
import numpy as np

# Perceptron input size:
input_size = 5

# Instantiating the perceptron:
perceptron = Neuron(num_inputs=input_size,
                    activation_function=sigmoid_function)

print("Perceptron's random weights = {}, and random bias = {}".format(
    perceptron.W, perceptron.b))

x = np.random.rand(input_size).reshape(1, input_size)
x = np.dot(x, 2)  # Producto punto de arreglo aleatorio x
print("Input vector : {}".format(x))

y = perceptron.forward(x)
print("Perceptron's output value given `x` : {}".format(y))