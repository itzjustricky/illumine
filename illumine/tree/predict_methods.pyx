"""
    Methods to optimize the prediction function
    of the lucid tree.

    @author: Ricky Chang
"""

import numpy as np
cimport numpy as cnp


cdef _find_activated_for_split(cnp.ndarray[double, ndim=2] X,
                               int col_index,
                               str relation,
                               float threshold):
    """ Return a numpy.ndarray of booleans that
        indicate whether or not X satisfies

    :param X: 2d numpy arary containing data to test condition on
    :param col_index: the index of the column to test the condition on
    :param relation: the string that determines the relation to be
        tested (i.e. [<, <=, >=, >])
    :param threshold: number that X elements will be tested against
    """

    if relation == '<=':
        return X[:, col_index] <= threshold
    elif relation == '>':
        return X[:, col_index] > threshold
    elif relation == '<':
        return X[:, col_index] < threshold
    elif relation == '>=':
        return X[:, col_index] >= threshold


def find_activated(X, leaf_path):
    """ Find the indices of the datapoints that satisfy
        a certain leaf_path

    :type X: np.ndarray
    :param X: a 2d matrix containing all the datapoints
        to be used to create predictions
    :param leaf_path (list or Iterable): list of tree-split objects
         that define the path for a terminal node
    """
    # cdef cnp.ndarray[bint, ndim=2] activation_matrix
    # cdef bint[:, :] activation_matrix
    activation_matrix = np.ones(X.shape, dtype=bool, order='F')

    # tree split is an object
    for tree_split in leaf_path:
        col_index = tree_split.feature

        activation_matrix[:, col_index] &= \
            _find_activated_for_split(
                X, col_index=col_index,
                relation=tree_split.relation,
                threshold=tree_split.threshold)

    return activation_matrix.all(axis=1)


def create_prediction(X_df, tuple leaf_paths, tuple leaf_values):
    """ Generate a prediction

    :param X_df: dataframe containing the dependent variables
    :param leaf_paths: list of illumine.woodland.LeafPath objects
    :param leaf_values: list of floats representing the values
        of the leaf nodes
    """
    # features integer representation
    X_matrix = X_df.values.astype(dtype=np.float64, order='F')

    y_pred = np.zeros(X_matrix.shape[0])
    for path, value in zip(leaf_paths, leaf_values):
        y_pred += find_activated(X_matrix, path) \
            * value

    return y_pred


def create_apply(X_df, tuple leaf_paths, tuple leaf_indices):
    """ Creates a nxm matrix of integers
        where n is the # of sample-data
              m is the # of estimators
        Note:
        Leaf indices are derived from the order in the
        preorder traversal of the decision tree.

        Each entry of the matrix gives an index
        of which leaf was activated in an estimator.
    """
    # features integer representation
    X_matrix = X_df.values.astype(dtype=np.float64, order='F')

    activated_indices = np.zeros(X_matrix.shape[0])
    for path, index in zip(leaf_paths, leaf_indices):
        activated_indices += \
            find_activated(X_matrix, path) * index

    return activated_indices
