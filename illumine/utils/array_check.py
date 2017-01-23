"""
    Description:



    @author: Ricky Chang
"""

import numbers

import numpy as np
import pandas as pd

__all__ = ['flatten_1darray']


def flatten_1darray(array):
    """ Convert the passed array to a flat numpy array
        Provides some checks on the array.

    :param array (pandas.Series|numpy.ndarray): should be
        a 1d array
    """
    if isinstance(array, pd.Series):
        return array.values
    elif isinstance(array, pd.DataFrame):
        return flatten_1darray(array.values)
    elif isinstance(array, np.ndarray):
        if array.ndim == 1:
            return array
        if array.ndim == 2:
            if array.shape[1] != 1:
                raise ValueError(' '.join((
                    "The passed array has shape {}, arrays".format(array.shape),
                    "with more than 2 dimensions are not supported"))
                )
            return array.ravel()
        else:
            raise ValueError("arrays with >2 dimensions are not supported")
    elif isinstance(array, list):
        if not all(map(lambda x: isinstance(x, numbers.Number))):
            raise ValueError("The passed array of type list should "
                             "only contain numeric types.")
        return np.array(array).ravel()
    else:
        raise ValueError(' '.join((
            "The passed array has an unsupported type {};"
            .format(type(array)),
            "only pd.Series, np.ndarrays and lists are supported"))
        )
