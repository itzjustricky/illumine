"""
    Description:
        Methods to optimize the prediction
        function of the lucid tree.

    @author: Ricky Chang
"""

import re

import numpy as np


# _inequality_pattern = re.compile('(<|<=|>=|>)')
_inequality_pattern = re.compile('(<=|>=|<|>)')


def find_activated_for_split(X, col_index, rel, thres):
    """ Return a numpy.ndarray of booleans that
        indicate whether or not X satisfies

    :param X (numpy.ndarray): 2d numpy arary containing
        data to test condition on
    :param col_index (int): the index of the column to
        test the condition on
    :param rel (string): the string that determines the
        relation to be tested (i.e. [<, <=, >=, >])
    :param thres (numeric): number that X elements will be
        tested against

    >>> X = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> col_index = 1
    >>> rel = '<='
    >>> thres = 8
    >>> determine_relation(X, int_repr, rel, thres)
    [True, True, True, False, False]
    """

    if rel == '<':
        return X[:, col_index] < thres
    elif rel == '<=':
        return X[:, col_index] <= thres
    elif rel == '>':
        return X[:, col_index] > thres
    elif rel == '>=':
        return X[:, col_index] >= thres


def find_activated(X, f_map, leaf_path):
    """ Find the indices of the datapoints that satisfy
        a certain leaf_path

    :param X (np.ndarray): a numpy 2d matrix containing all
        the datapoints to be used to create predictions
    :param f_map (dict): the feature to int mapping
    :param leaf_path (list or Iterable): list of splits
         that define the path for a terminal node
    """
    condition_matrix = np.ones(X.shape, dtype=bool)

    for tree_split in leaf_path:
        # parse the tree_split string first
        feature_name, rel, thres = \
            _inequality_pattern.split(tree_split)
        thres = float(thres)
        f_int_repr = f_map[feature_name]

        condition_matrix[:, f_int_repr] &= \
            find_activated_for_split(X, f_int_repr, rel, thres)

    return np.all(condition_matrix, axis=1)


def map_features_to_int(feature_names):
    f_map = dict()
    for ind, feature in enumerate(feature_names):
        f_map[feature] = ind

    return f_map


def create_prediction(X_df, leaf_paths, leaf_values):

    # features integer representation
    f_map = map_features_to_int(X_df.columns)
    X_matrix = X_df.values

    y_pred = np.zeros(X_matrix.shape[0])
    for path, value in zip(leaf_paths, leaf_values):
        y_pred += find_activated(X_matrix, f_map, path) \
            * value

    return y_pred


if __name__ == "__main__":
    pass
