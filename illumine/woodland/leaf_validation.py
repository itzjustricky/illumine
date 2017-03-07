"""
    Description:
        Module containing functions to validate
        leaves from a LucidSKEnsemble object.


    @author: Ricky Chang
"""

import logging
from collections import Iterable

import numpy as np

from .leaf_objects import LeafPath
from .leaf_objects import CompressedEnsemble

from .predict_methods import find_activated


__all__ = ['score_leaves', 'score_leaf_group']


def score_leaves(compressed_ensemble, X_df, y_true,
                 score_function, required_threshold=0,
                 considered_paths=None, normalize_score=False):
    """ Score leaves based on some passed score_function.
        The score will be calculated only over the data-points
        where the leaf is activated.

        The y_true value will be adjusted by any init_estimator
        that the compressed_ensemble has.

    :param compressed_ensemble (CompressedEnsemble): a CompressedEnsemble
        object used to extract leaves
    :param X_df (pandas.DataFrame): the X matrix to score the leaves on
    :param y_true (array-like type): the y values which the X matrix will
        be tested against
    :param score_function (function): function used to calculate score with
        function signature of score(X, y)
    :param considered_paths (array-like type): a list of SKTreeNodLeafPath; only
        the LeafPath in considered_paths will be considered. Defaults
        to None, if None then all leaves in compressed_ensemble will be considered.
    :param required_threshold (int): if a leaf is activated less than the
        required_threshold # of times then it will be given a rank of -inf
    :param normalize_score (bool): indicates whether or not to normalize
        the score by the # of activated indices for a certain leaf
    """
    if not isinstance(compressed_ensemble, CompressedEnsemble):
        raise ValueError(
            "The passed argument compressed_ensemble should "
            "be an instance of CompressedEnsemble")

    X = X_df.values
    if considered_paths is not None:
        if not all(map(lambda x: isinstance(x, LeafPath), considered_paths)):
            raise ValueError(
                "All elements of considered_paths should be of type LeafPath.")
        filtered_leaves = \
            dict(((path, compressed_ensemble[path])
                  for path in considered_paths))
    else:
        filtered_leaves = compressed_ensemble

    if not isinstance(y_true, Iterable):
        raise ValueError("The passed y_true argument is not iterable.")
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    y_adjusted = y_true.ravel() - compressed_ensemble._init_estimator.predict(X_df).ravel()

    scores = dict()
    for ind, leaf_pair in enumerate(filtered_leaves.items()):
        path, value = leaf_pair

        activated_indices = find_activated(X, path)
        n_activated = np.sum(activated_indices)
        if np.sum(activated_indices) < required_threshold:
            scores[path] = -np.inf
        else:
            y1 = value * np.ones(n_activated)
            y2 = y_adjusted[np.where(activated_indices)[0]].ravel()
            if y1.shape != y2.shape:
                raise ValueError(
                    "The passed y_true argument should "
                    "be a 1-dimensional array.")
            scores[path] = score_function(y1, y2)

            if normalize_score:
                scores[path] /= n_activated

    return scores


def score_leaf_group(leaf_group, compressed_ensemble,
                     X_df, y_true, score_function,
                     required_threshold=0, normalize_score=False):
    """ Score leaves based on some passed score_function.
        The score will be calculated only over the data-points
        where the leaf is activated.

        The y_true value will be adjusted by any init_estimator
        that the compressed_ensemble has.

    :param leaf_group (array-like type): a list of SKTreeNodLeafPaths
        The scoring will be done as a cumulative of the leaves in
        the leaf_group.
    :param compressed_ensemble (CompressedEnsemble): a CompressedEnsemble
        object used to extract leaves
    :param X_df (pandas.DataFrame): the X matrix to score the leaves on
    :param y_true (array-like type): the y values which the X matrix will
        be tested against
    :param score_function (function): function used to calculate score with
        function signature of score(y_pred, y_true)
    :param required_threshold (int): if a leaf is activated less than the
        required_threshold # of times then it will be given a rank of -inf
    :param normalize_score (bool): indicates whether or not to normalize
        the score by the # of activated indices for a certain leaf
    """

    if not isinstance(compressed_ensemble, CompressedEnsemble):
        raise ValueError(
            "The passed argument compressed_ensemble should "
            "be an instance of CompressedEnsemble")
    X = X_df.values

    filtered_leaves = \
        dict(((path, compressed_ensemble[path])
              for path in leaf_group))

    accm_value = 0
    activated_indices = np.ones(X.shape[0], dtype=bool)
    for ind, leaf_pair in enumerate(filtered_leaves.items()):
        path, value = leaf_pair
        accm_value += value
        activated_indices &= find_activated(X, path)

    n_activated = np.sum(activated_indices)
    logging.getLogger(__name__).debug(
        "The # of activated indices is {}".format(n_activated))

    if not isinstance(y_true, Iterable):
        raise ValueError("The passed y_true argument is not iterable.")
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    y_adjusted = y_true.ravel() - compressed_ensemble._init_estimator.predict(X_df).ravel()

    if n_activated < required_threshold:
        group_score = -np.inf
    else:
        y1 = accm_value * np.ones(n_activated)
        y2 = y_adjusted[np.where(activated_indices)[0]].ravel()
        if y1.shape != y2.shape:
            raise ValueError(
                "The passed y_true argument should "
                "be a 1-dimensional array.")
        group_score = score_function(y1, y2)

        if normalize_score:
            group_score /= n_activated

    return group_score
