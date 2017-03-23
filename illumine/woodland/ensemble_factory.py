"""
    The functions in this module are used to create
    the objects in the leaf_objects.py module.

     Notes:
        The hash of an object depends on the print_precision
        since the hash is of the str of path of the LucidTree

    @author: Ricky
"""

from collections import Iterable

import numpy as np
from sklearn.dummy import DummyRegressor

from .lucid_ensemble import LucidEnsemble
from ..tree.tree_factory import make_many_LucidTrees
# from ..tree.lucid_tree import LucidTree
# from ..tree.tree_factory import assemble_lucid_trees


__all__ = ['make_LucidEnsemble']


def make_LucidEnsemble(sk_ensemble, feature_names, print_precision=5,
                       init_estimator=None, ensemble_kws=None):
    """ Breakdown a tree's splits and returns the value of every leaf along
        with the path of splits that led to the leaf

    ..note:
        Scikit-learn represent their trees with nodes (represented by numbers) printed
        by preorder-traversal; number of -2 represents a leaf, the other numbers are by
        the index of the column for the feature

    :param sk_ensemble: scikit-learn ensemble model object
    :param feature_names (list): list of names (strings) of the features that
        were used to split the tree
    :param init_estimator (function): the initial estimator of the ensemble
        defaults to None, if None then equals Scikit-learn tree's initial estimator
    :param print_precision (int): to determine what number the node
        values, thresholds are rounded to when printing
    :param tree_kws (dict): key-word arguments to be passed into
        LucidTree's constructor
    :param ensemble_kws (dict): key-word arguments to be passed into
        LucidEnsemble's constructor

    :returns: dictionary of TreeNode objects indexed by their order in the
        pre-order traversal of the Decision Tree
    """
    if ensemble_kws is None:
        ensemble_kws = dict()
    elif not isinstance(ensemble_kws, dict):
        raise ValueError("tree_kws should be of type dict")

    if isinstance(sk_ensemble.estimators_, np.ndarray):
        ensemble_estimators = sk_ensemble.estimators_.ravel()
    elif isinstance(sk_ensemble.estimators_, Iterable):
        ensemble_estimators = sk_ensemble.estimators_

    lucid_trees = make_many_LucidTrees(
        ensemble_estimators, feature_names, print_precision)

    # Retrieve the initial estimator for the ensemble
    if init_estimator is None:
        try:
            init_estimator = sk_ensemble.init_
        except AttributeError:
            init_estimator = DummyRegressor(constant=0.0)
            init_estimator.fit(np.zeros((10, len(feature_names))), np.zeros(10))
    elif not hasattr(init_estimator, 'predict'):
        raise ValueError(
            "The init_estimator should be an object with a predict "
            "function with function signature predict(self, X) "
            "where X is the feature matrix.")

    try:
        learning_rate = sk_ensemble.learning_rate
    except AttributeError:
        learning_rate = 1.0 / sk_ensemble.n_estimators

    return LucidEnsemble(
        lucid_trees,
        init_estimator=init_estimator,
        learning_rate=learning_rate,
        **ensemble_kws)
