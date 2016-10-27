"""
    Some simple utility functions to help with statistical
    computations

"""


import numpy as np


def get_gaussian_bounds(x_vect, critical_value=1.96):
    """
    :returns: a tuple of the outlier bounds
    """

    x_mean = np.mean(x_vect)
    x_stdev = np.std(x_vect)
    return (x_mean - critical_value * x_stdev,
            x_mean + critical_value * x_stdev)
