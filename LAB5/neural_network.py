"""
Neural Network class

Created on December 18th 2020

@authors: Niv Ben Ami & Ziv Zango
"""
import numpy


class NeuralNetwork:

    # initialize variables in class
    def __init__(self, inputs, outputs, layer_nodes):
        # initialize inputs, outputs and error for training
        self.inputs = inputs
        self.outputs = outputs
        self.error = None
        # initialize layers
        self.h_layer1 = None
        self.o_layer = None
        # initialize weights
        self.weights0 = numpy.random.normal(0, 1, (self.inputs.shape[1], layer_nodes))
        self.weights1 = numpy.random.normal(0, 1, (layer_nodes, self.outputs.shape[1]))
        # initialize biases
        self.bias0 = numpy.random.normal(0, 1, layer_nodes)
        self.bias1 = numpy.random.normal(0, 1)
        self.error_history = []
        self.epoch_list = []

    # activation function ==> S(x) = 1/1+e^(-x)
    @staticmethod
    def sigmoid(x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + numpy.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.h_layer1 = self.sigmoid(numpy.dot(self.inputs, self.weights0) + self.bias0)
        self.o_layer = self.sigmoid(numpy.dot(self.h_layer1, self.weights1) + self.bias1)

    # going backwards through the network to update weights
    def back_propagation(self, learning_rate):
        self.error = self.outputs - self.o_layer

        delta1 = self.error * self.sigmoid(self.o_layer, deriv=True)
        self.weights1 += learning_rate * numpy.dot(self.h_layer1.T, delta1)

        delta0 = numpy.dot(self.error, self.weights1.T) * self.sigmoid(self.h_layer1, deriv=True)
        self.weights0 += learning_rate * numpy.dot(self.inputs.T, delta0)

        # Updating the bias weight value
        for b in delta1:
            self.bias1 += learning_rate * b
        for b in delta0:
            self.bias0 += learning_rate * b

    # train the neural net for 2,000 iterations
    def train(self, epochs=2000, learning_rate=2):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.back_propagation(learning_rate)
            # keep track of the error history over each epoch
            self.error_history.append(numpy.average(numpy.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        self.h_layer1 = self.sigmoid(numpy.dot(new_input, self.weights0) + self.bias0)
        self.o_layer = self.sigmoid(numpy.dot(self.h_layer1, self.weights1) + self.bias1)
        return self.o_layer
