"""
LMS/NLMS filters class

Created on November 20th 2020

@authors: Niv Ben Ami & Ziv Zango
"""

import numpy


class LMSFilter:
    """
    A class to perform LMS/NLMS adaptive filters

    Attributes
    ----------
    u : array-like
        One-dimensional filter input.
    d : array-like
        One-dimensional desired signal, i.e., the output of the unknown FIR
        system which the adaptive filter should identify.
        Must have length >=len(u).
    filter_length : int
        Desired number of filter taps (desired filter order + 1), must be
        positive.
    step_size : float
        Step size of the algorithm also called "Learnig rate", must be positive.
        in the case there is no interference the step size for NLMS algorithm
        is 1.

    Raises
    ------
    TypeError
        If number of filter taps filter_length is not type integer,
        or if step_size is not type float/int.
    ValueError
        If number of filter taps filter_length is negative,
        or if step_size is negative.

    Methods
    -------
    _check_input()
        PRIVATE METHOD: checks the inputs are set correctly
    regular(w=None)
        performing LMS adaptive filter
    normalized(w=None)
        performing NLMS adaptive filter
    """

    def __init__(self, u, d, filter_length, step_size):
        self._mu = step_size
        self._m = filter_length
        self._check_input()
        self._u = u
        self._d = d
        self._n = len(u) - self._m + 1

    def _check_input(self):
        if type(self._mu) is not float and type(self._mu) is not int:
            raise TypeError('Step size must be type float (or integer)')
        elif self._mu < 0:
            raise ValueError('Step size must non-negative')
        if type(self._m) is not int:
            raise TypeError('Number of filter taps must be type integer')
        elif self._m <= 0:
            raise ValueError('Length of filter taps must be greater than 0')

    def regular(self, w=None):
        """
        Perform least-mean-squares (LMS) adaptive filtering on _u to minimize error
        given by e=d-y, where y is the output of the adaptive filter.

        Optional Parameters
        -------------------
        w : array-like
            One-dimensional of the filter coefficients. Should match desired number of
            filter taps, defaults to zeros.

        Returns
        -------
        y : numpy.array
            Output values of LMS filter, array of length _n of the class.
        e : numpy.array
            Error signal, i.e, d-y. Array of length _n of the class.
        w : numpy.array
            Final filter coefficients in array with the length filter_length of the class
        """

        # Initialization
        if w is None:
            w = numpy.zeros(self._m)
        else:
            self._m = len(w)
        e = numpy.zeros(self._n)
        y = numpy.zeros(self._n)

        # Perform filtering with loop
        for index in range(self._n):
            x = self._u[index:index + self._m]
            y[index] = numpy.dot(x, w)
            e[index] = self._d[index + self._m - 1] - y[index]

            # Updating filter coefficients
            w = w + self._mu * e[index] * x
            y[index] = numpy.dot(x, w)
        return y, e, w

    def normalized(self, w=None):
        """
        Perform normalized least-mean-squares (NLMS) adaptive filtering on u to
        minimize error given by e=d-y, where y is the output of the adaptive
        filter.

        Optional Parameters
        -------------------
        w : array-like
            One-dimensional of the filter coefficients. Should match desired number of
            filter taps, defaults to zeros.

        Returns
        -------
        y : numpy.array
            Output values of NLMS filter, array of length _n of the class.
        e : numpy.array
            Error signal, i.e, d-y. Array of length _n of the class.
        w : numpy.array
            Final filter coefficients in array with the length filter_length of the class
        """

        # Initialization
        if w is None:
            w = numpy.zeros(self._m)
        else:
            self._m = len(w)
        e = numpy.zeros(self._n)
        y = numpy.zeros(self._n)

        # Perform filtering with loop
        for index in range(self._n):
            x = self._u[index:index + self._m]
            y[index] = numpy.dot(x, w)
            e[index] = self._d[index + self._m - 1] - y[index]

            # Updating filter coefficients
            w += self._mu * e[index] * x / numpy.dot(x, x)
            y[index] = numpy.dot(x, w)
        return y, e, w
