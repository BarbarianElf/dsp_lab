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
from neural_network import NeuralNetwork as NN

MAX_INPUT = 8


def get_bits_vectors(number_of_bits):
    """
    Generate array-like of arrays with the shape(index, bit_vector)
    with all the bit combination with the length of number_of_bits

    Parameters
    ----------
    number_of_bits : int
        number of bits in the vector

    Returns
    -------
    : array-like of array
        array with all bit combination vectors
        with the shape (2**number_of_bits, number_of_bits)
    """
    bits_vectors = numpy.reshape(numpy.arange(0, 2**number_of_bits, dtype=numpy.uint8), (-1, 1))
    return numpy.unpackbits(bits_vectors, axis=1, bitorder='little', count=number_of_bits)


def get_operation_bit_vector_output(x, operation=operator.xor):
    """
    Generate array-like of arrays with the shape(index, bit_vector)
    with all the bit combination with the length of number_of_bits

    Parameters
    ----------
    x : array-like of array
        array of bit combination vectors

    Optional Parameters
    -------------------
    operation : operator-type
        operator between the bits, default xor.
        for more information
        please read operator.py (python built in)

    Returns
    -------
    : array-like of array
        array with output of the operator for all bits
        with the shape (2**number_of_bits, output)
    """
    return numpy.reshape(numpy.asarray([reduce(operation, x[index]) for index in range(MAX_INPUT)]), (-1, 1))


if __name__ == "__main__":
    plt = lu.get_plt()
    os.makedirs("results", exist_ok=True)

    # LAB 5
    x_input = get_bits_vectors(int(numpy.log2(MAX_INPUT)))
    y_output = get_operation_bit_vector_output(x_input)

    iteration = 100
    hidden_layer_nodes = 3

    for q in range(1, 3):
        hidden_layer_nodes *= q
        t = numpy.arange(1, 101)
        error = []
        for i in range(1, iteration + 1):
            n_net = NN(x_input, y_output, hidden_layer_nodes)
            n_net.train(epochs=2000, learning_rate=2)
            error.append(numpy.square(n_net.error).mean())
        plt.figure(q)
        plt.title(f'MSE over {iteration} iterations\nHidden layer with {hidden_layer_nodes} nodes')
        plt.xlabel('iteration')
        plt.ylabel('MSE')
        plt.xlim([1, iteration])
        plt.plot(t, error)
        plt.grid(True)
        plt.savefig(f"results/MSE_for_{hidden_layer_nodes}_nodes")

    # Unmark the command blow to show ALL the figures
    # plt.show()
