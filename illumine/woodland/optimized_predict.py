"""
    Description:
        Methods to optimize the prediction
        function of the lucid tree.

    TODO:
        I should be able to easily move parts of
        the function to Cython to make it even faster

    @author: Ricky Chang
"""

import numpy as np


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

    # is_close = np.isclose(X[:, col_index], thres)

    # a bitwise operations are needed to deal
    # with float precision issues
    if rel == '<=':
        return X[:, col_index] <= thres
        """
        return np.bitwise_or(
            X[:, col_index] <= thres, is_close)
        """
    elif rel == '>':
        return X[:, col_index] > thres
        """
        return np.bitwise_and(
            X[:, col_index] > thres,
            np.bitwise_xor(X[:, col_index] > thres, is_close)
        )
        """
    if rel == '<':
        return X[:, col_index] < thres
        """
        return np.bitwise_and(
            X[:, col_index] < thres,
            np.bitwise_xor(X[:, col_index] < thres, is_close)
        )
        """
    elif rel == '>=':
        return X[:, col_index] >= thres
        """
        return np.bitwise_or(
            X[:, col_index] >= thres, is_close)
        """


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
        col_index = f_map[tree_split.feature_name]

        condition_matrix[:, col_index] &= \
            find_activated_for_split(
                X, col_index=col_index,
                rel=tree_split.relation,
                thres=tree_split.threshold)

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
