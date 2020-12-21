"""
lab 5 for DSP LAB

Created on December 18th 2020

@authors: Niv Ben Ami & Ziv Zango
"""
import numpy
import os

import operator
from functools import reduce

import lab_utils as lu
from LAB5.neural_network import NeuralNetwork as NN

MAX_INPUT = 8


def get_system_input(number_of_bits):
    system_input = numpy.reshape(numpy.arange(0, 2**number_of_bits, dtype=numpy.uint8), (-1, 1))
    return numpy.unpackbits(system_input, axis=1, bitorder='little', count=number_of_bits)


def get_system_output(x, operation=operator.xor):
    return numpy.reshape(numpy.asarray([reduce(operation, x[i]) for i in range(MAX_INPUT)]), (-1, 1))


if __name__ == "__main__":
    plt = lu.get_plt()

    x_input = get_system_input(int(numpy.log2(MAX_INPUT)))
    y_output = get_system_output(x_input)

    iteration = 100
    hidden_layer_nodes = 3

    for q in range(1, 3):
        hidden_layer_nodes *= q
        error = []
        t = []
        for i in range(iteration):
            n_net = NN(x_input, y_output, hidden_layer_nodes)
            n_net.train(epochs=2000)
            error.append(numpy.square(n_net.error).mean())
            t.append(i)
        plt.figure(q)
        plt.title(f'MSE over {iteration} iterations\nwith {hidden_layer_nodes} nodes (Hidden layer)')
        plt.xlabel('iteration')
        plt.ylabel('MSE')
        plt.ylim([0, 0.5])
        plt.plot(t, error)
        plt.grid(True)
    plt.show()

