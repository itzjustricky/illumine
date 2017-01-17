"""
    Description:
        Methods to optimize the prediction
        function of the lucid tree.

    @author: Ricky Chang
"""

import numpy as np


def _map_features_to_int(feature_names):
    f_map = dict()
    for ind, feature in enumerate(feature_names):
        f_map[feature] = ind

    return f_map


def _find_activated_for_split(X, col_index, relation, threshold):
    """ Return a numpy.ndarray of booleans that
        indicate whether or not X satisfies

    :param X (numpy.ndarray): 2d numpy arary containing
        data to test condition on
    :param col_index (int): the index of the column to
        test the condition on
    :param relation (string): the string that determines the
        relation to be tested (i.e. [<, <=, >=, >])
    :param threshold (numeric): number that X elements will be
        tested against

    >>> X = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> col_index = 1
    >>> relation = '<='
    >>> threshold = 8
    >>> _find_activated_for_split(X, col_index, relation, threshold)
    [True, True, True, False, False]
    """

    if relation == '<=':
        return X[:, col_index] <= threshold
    elif relation == '>':
        return X[:, col_index] > threshold
    if relation == '<':
        return X[:, col_index] < threshold
    elif relation == '>=':
        return X[:, col_index] >= threshold


def _find_activated(X, f_map, leaf_path):
    """ Find the indices of the datapoints that satisfy
        a certain leaf_path

    :param X (np.ndarray): a numpy 2d matrix containing all
        the datapoints to be used to create predictions
    :param f_map (dict): the feature to int mapping
    :param leaf_path (list or Iterable): list of splits
         that define the path for a terminal node
    """
    activation_matrix = np.ones(X.shape, dtype=bool)

    for tree_split in leaf_path:
        col_index = f_map[tree_split.feature_name]

        activation_matrix[:, col_index] &= \
            _find_activated_for_split(
                X, col_index=col_index,
                relation=tree_split.relation,
                threshold=tree_split.threshold)

    return np.all(activation_matrix, axis=1)


def create_prediction(X_df, leaf_paths, leaf_values):

    # features integer representation
    f_map = _map_features_to_int(X_df.columns)
    X_matrix = X_df.values

    y_pred = np.zeros(X_matrix.shape[0])
    for path, value in zip(leaf_paths, leaf_values):
        y_pred += _find_activated(X_matrix, f_map, path) \
            * value

    return y_pred


def create_apply(X_df, leaf_paths, leaf_indices):
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
    f_map = _map_features_to_int(X_df.columns)
    X_matrix = X_df.values

    activated_indices = np.zeros(X_matrix.shape[0])
    for path, index in zip(leaf_paths, leaf_indices):
        activated_indices += \
            _find_activated(X_matrix, f_map, path) * index

    return activated_indices
