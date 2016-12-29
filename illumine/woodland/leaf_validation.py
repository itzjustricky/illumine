"""
    Description:
        Module containing functions to validate
        leaves from a LucidSKEnsemble object.


    @author: Ricky Chang
"""

import numpy as np

from .leaf_objects import LucidSKEnsemble
from .leaf_objects import SKTreeNode

from .optimized_predict import map_features_to_int
from .optimized_predict import find_activated


def score_leaves(lucid_ensemble, X_df, y_true,
                 score_function, required_threshold=0,
                 considered_leaves=None, normalize_score=False):
    """ Score leaves based on some passed score_function

    :param lucid_ensemble (LucidSKEnsemble): a compressed LucidSKEnsemble
        object used to extract leaves and
    :param X_df (pandas.DataFrame): the X matrix to score the leaves on
    :param y_true (array-like type): the y values which the X matrix will
        be tested against
    :param score_function (function): function used to calculate score with
        function signature of score(X, y)
    :param considered_leaves (array-like type): a list of SKTreeNode; only
        the SKTreeNodes in considered_leaves will be considered. Defaults
        to None, if None then all leaves in lucid_ensemble will be considered.
    :param required_threshold (int): if a leaf is activated less than the
        required_threshold # of times then it will be given a rank of -inf
    :param normalize_score (bool): indicates whether or not to normalize
        the score by the # of activated indices for a certain leaf
    """
    if not isinstance(lucid_ensemble, LucidSKEnsemble):
        raise ValueError(
            "The passed argument lucid_ensemble should "
            "be an instance of LucidSKEnsemble")

    f_map = map_features_to_int(X_df.columns)
    X = X_df.values

    if considered_leaves is not None:
        if not all(map(lambda x: isinstance(x, SKTreeNode), considered_leaves)):
            raise ValueError(
                "All elements of considered_leaves should be of type SKTreeNode.")
        filtered_leaves = \
            dict(((key, lucid_ensemble.compressed_ensemble[key])
                  for key in considered_leaves))
    else:
        filtered_leaves = lucid_ensemble.compressed_ensemble

    scores = dict()
    for ind, leaf_pair in enumerate(filtered_leaves.items()):
        path, value = leaf_pair
        activated_indices = find_activated(X, f_map, path.split(' & '))
        n_activated = np.sum(activated_indices)
        if np.sum(activated_indices) < required_threshold:
            scores[path] = -np.inf
        else:
            scores[path] = score_function(
                value * np.ones(n_activated),
                y_true[np.where(activated_indices)[0]])
            if normalize_score:
                scores[path] /= n_activated

    return scores
