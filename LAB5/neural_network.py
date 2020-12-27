"""
Neural Network class

Created on December 18th 2020

@authors: Niv Ben Ami & Ziv Zango
"""
import numpy


class NeuralNetwork:
    """
    A class of Artificial Neural Network with 1 hidden layer

    Attributes
    ----------
    inputs : array-like of arrays
        Two-dimensional input (index, input) for training,
        pairing with output.
    outputs : array-like of arrays
        Two-dimensional output (index, output) for training,
        pairing with input.
    layer_nodes : int
        Desired number of neurals in hidden layer, must be natural.

    Raises
    ------
    TypeError
        If number of neurals layer_nodes is not type integer.
    ValueError
        If number of of neurals layer_nodes is negative.

    Methods
    -------
    _check_input()
        PRIVATE METHOD: checks the inputs are set correctly
    sigmoid(x)
        activation function
    feed_forward()
        feeding the Neural Network
    back_propagation(learning_rate)
        update weights of the Neural Network
    train(epochs=10000, learning_rate=1)
        train the Neural Network epochs times
    predict(new_input)
        predict the output for new_input
    """

    def __init__(self, inputs, outputs=None, layer_nodes=1):
        # initialize inputs, outputs and error for training
        self.inputs = inputs
        self.outputs = outputs
        self.error = None
        self._check_input(layer_nodes)
        # initialize layers
        self.h_layer1 = None
        self.o_layer = None
        # initialize weights
        self.weights0 = numpy.random.normal(0, 1, (self.inputs.shape[1], layer_nodes))
        self.weights1 = numpy.random.normal(0, 1, (layer_nodes, self.outputs.shape[1]))
        # initialize biases
        self.bias0 = numpy.random.normal(0, 1, layer_nodes)
        self.bias1 = numpy.random.normal(0, 1)
        # initialize tracking arrays
        self.error_history = []
        self.epoch_list = []

    def _check_input(self, neurals):
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError('Number of input vectors must be the same as output vectors')
        if type(neurals) is not int:
            raise TypeError('Number of neurals must be type integer')
        elif neurals <= 0:
            raise ValueError('Number of neurals must be greater than 0')

    @staticmethod
    def sigmoid(x, deriv=False):
        """
        Perform sigmoid function S(x) = 1/1+e^(-x)
        as activation function of the Neural Network

        Parameters
        ----------
        x : int, array-like..

        Optional Parameters
        -------------------
        deriv : boolean
            boolean condition for return the derivative,
            defaults to False.

        Returns
        -------
        :float, array-like..
            Calculation of sigmoid function,
            or the derivative of sigmoid function
        """
        if deriv:
            return x * (1 - x)
        return 1 / (1 + numpy.exp(-x))

    def feed_forward(self, new_input=None):
        """
        Feed forward the input data through the Neural Network,
        and produce the output.
        """
        if new_input is None:
            i_layer = self.inputs
        else:
            i_layer = new_input
        self.h_layer1 = self.sigmoid(numpy.dot(i_layer, self.weights0) + self.bias0)
        self.o_layer = self.sigmoid(numpy.dot(self.h_layer1, self.weights1) + self.bias1)

    def back_propagation(self, learning_rate):
        """
        Set the error based the desired output and output layer.
        Going backwards through the network to make corrections
        (update weights and bias values), based on the error values.

        Parameters
        ----------
        learning_rate : float
            learning rate of the Neural Network.
        """
        self.error = self.outputs - self.o_layer

        delta1 = self.error * self.sigmoid(self.o_layer, deriv=True)
        delta0 = numpy.dot(delta1, self.weights1.T) * self.sigmoid(self.h_layer1, deriv=True)

        self.weights1 += learning_rate * numpy.dot(self.h_layer1.T, delta1) / delta1.shape[0]
        self.weights0 += learning_rate * numpy.dot(self.inputs.T, delta0) / delta0.shape[0]

        self.bias1 += learning_rate * delta1.mean()
        self.bias0 += learning_rate * delta0.mean(axis=0)

    def train(self, epochs=10000, learning_rate=1):
        """
        Perform training of the Neural Network epochs times

        Optional Parameters
        -------------------
        epochs : int
            number of training iteration, default 10,000.
        learning_rate: float
            set learning rate of the Neural Network,
            default 1.
        """
        for epoch in range(epochs):
            self.feed_forward()
            self.back_propagation(learning_rate)

            # keep track of the error history over each epoch
            self.error_history.append(numpy.average(numpy.abs(self.error)))
            self.epoch_list.append(epoch)

    def predict(self, new_input):
        """
        Predict output on the new input data via the Neural Network

        Parameters
        ----------
        new_input : array-like (N-dimensional)
            N-dimensional input (index, input) to predict the output.

        Returns
        -------
        y : array-like (N-dimensional)
            N-dimensional output (index, output) of the Neural Network,
            i.e the prediction.
        """
        self.feed_forward(new_input)
        return self.o_layer
