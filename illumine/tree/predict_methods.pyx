"""
    Methods to optimize the prediction function
    of the lucid tree.

    @author: Ricky Chang
"""

cimport cython

import numpy as np
cimport numpy as cnp
from pandas import DataFrame


cdef _find_activated_for_split(cnp.ndarray[double, ndim=1] x_array,
                               str relation,
                               float threshold):
    """ Return a numpy.ndarray of booleans that
        indicate whether or not X satisfies

    :param X: 2d numpy arary containing data to test condition on
    :param relation: the string that determines the relation to be
        tested (i.e. [<, <=, >=, >])
    :param threshold: number that X elements will be tested against
    """

    if relation == '<=':
        return x_array <= threshold
    elif relation == '>':
        return x_array > threshold
    elif relation == '<':
        return x_array < threshold
    elif relation == '>=':
        return x_array >= threshold


@cython.cdivision(True)
def find_activated(X, leaf_path):
    """ Find the indices of the datapoints that satisfy
        a certain leaf_path

    :type X: np.ndarray
    :type leaf_path: TreeSplit Object
    :param X: a 2d matrix containing all the datapoints to be used to
        create predictions
    :param leaf_path: object that defines the path for a terminal node
    """
    # cdef cnp.ndarray[bint, ndim=2] activation_matrix
    # cdef bint[:, :] activation_matrix
    activation_matrix = np.ones(X.shape, dtype=bool, order='F')

    cdef int col_index
    # tree split is an object
    for tree_split in leaf_path:
        col_index = tree_split.feature

        activation_matrix[:, col_index] &= \
            _find_activated_for_split(
                X[:, col_index],
                relation=tree_split.relation,
                threshold=tree_split.threshold)

    return activation_matrix.all(axis=1)


@cython.cdivision(True)
def create_prediction(cnp.ndarray[double, ndim=2] X, tuple leaf_paths, tuple leaf_values):
    """ Generate a prediction

    :param X: matrix containing the dependent variables
    :param leaf_paths: list of illumine.woodland.LeafPath objects
    :param leaf_values: list of floats representing the values
        of the leaf nodes
    """
    y_pred = np.zeros(X.shape[0])
    for path, value in zip(leaf_paths, leaf_values):
        y_pred += find_activated(X, path) * value

    return y_pred


@cython.cdivision(True)
def create_apply(cnp.ndarray[double, ndim=2] X, tuple leaf_paths, tuple leaf_indices):
    """ Creates a nxm matrix of integers
        where n is the # of sample-data
              m is the # of estimators
        Note:
        Leaf indices are derived from the order in the
        preorder traversal of the decision tree.

        Each entry of the matrix gives an index
        of which leaf was activated in an estimator.
    """
    activated_indices = np.zeros(X.shape[0])
    for path, index in zip(leaf_paths, leaf_indices):
        activated_indices += find_activated(X, path) * index

    return activated_indices
