"""
    Description:
        The functions in this module are used to create
        the objects in the leaf_objects.py module.

     make_LucidSKTree : factory method for LucidSKTree;
     make_LucidSKEnsemble : factory method for LucidSKEnsemble;
     unique_leaves_per_sample : function
     get_tree_predictions : function

     Notes:
        The hash of an object depends on the print_precision
        since the hash is of the str of path of the LucidSKTree

    TODO:
        * The make_LucidSKTree function may be better inside
            the LucidSKTree __init__ function. As well as for
            LucidSKEnsemble.
            - problems .. it makes the __reduce__ process
                less clean

    @author: Ricky
"""

from collections import OrderedDict
from collections import Iterable

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

from .leaf_objects import SKTreeNode
from .leaf_objects import LucidSKTree
from .leaf_objects import LucidSKEnsemble

from ._retrieve_leaf_paths import retrieve_tree_metas

__all__ = ['make_LucidSKTree', 'make_LucidSKEnsemble']


def assemble_lucid_trees(sk_trees, feature_names, print_precision, **tree_kw):
    """ This function is used to gather the paths to all the
        leaves given some of the Scikit-learn attributes
        of a decision tree.
    """
    tree_metas = retrieve_tree_metas(
        *_accumulate_tree_attributes(sk_trees),
        # must be changed to strings to be passed into Cython function
        feature_names=list(map(str, feature_names)),
        print_precision=print_precision)

    lucid_trees = []
    for tree_meta in tree_metas:
        tree_leaves = OrderedDict()
        for leaf_meta in tree_meta:
            leaf_ind = leaf_meta[0]
            tree_leaves[leaf_ind] = SKTreeNode(*leaf_meta[1:])

        lucid_trees.append(
            LucidSKTree(tree_leaves, feature_names, **tree_kw))

    return lucid_trees


def _accumulate_tree_attributes(sk_trees):
    if not isinstance(sk_trees, Iterable):
        sk_trees = [sk_trees]

    accm_values = []
    accm_node_samples = []
    accm_features = []
    accm_thresholds = []

    for sk_tree in sk_trees:
        accm_values.append(sk_tree.tree_.value)
        accm_node_samples.append(sk_tree.tree_.n_node_samples)
        accm_features.append(sk_tree.tree_.feature)
        accm_thresholds.append(sk_tree.tree_.threshold)
    return (
        accm_values,
        accm_node_samples,
        accm_features,
        accm_thresholds,
    )


def make_LucidSKTree(sk_tree, feature_names, print_precision=10,
                     sort_by_index=True, tree_kw=None):
    """ Breakdown a tree's splits and returns the value of every leaf along
        with the path of splits that led to the leaf

    ..note:
        Scikit-learn represent their trees with nodes (represented by numbers)
        printed by preorder-traversal; number of -2 represents a leaf, the other
        numbers are by the index of the column for the feature

    :param feature_names (list): list of names (strings) of the features that
        were used to split the tree
    :param sk_tree: scikit-learn tree object
    :param print_precision (int): to determine what number the node
        values, thresholds are rounded to when printing
    :param tree_kw (dict): key-word arguments to be passed into
        LucidSKTree's constructor

    :returns: LucidSKTree object indexed by their order in the
        pre-order traversal of the Decision Tree
    """
    if tree_kw is None:
        tree_kw = dict()
    elif not isinstance(tree_kw, dict):
        raise ValueError("tree_kw should be of type dict")
    return assemble_lucid_trees(
        sk_tree, feature_names, print_precision, **tree_kw)[0]


def make_LucidSKEnsemble(sk_ensemble, feature_names, print_precision=10,
                         init_estimator=None, tree_kw=None,
                         ensemble_kw=None, **kwargs):
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
    :param tree_kw (dict): key-word arguments to be passed into
        LucidSKTree's constructor
    :param ensemble_kw (dict): key-word arguments to be passed into
        LucidSKEnsemble's constructor

    :returns: dictionary of SKTreeNode objects indexed by their order in the
        pre-order traversal of the Decision Tree
    """
    if tree_kw is None:
        tree_kw = dict()
    elif not isinstance(tree_kw, dict):
        raise ValueError("tree_kw should be of type dict")
    if ensemble_kw is None:
        ensemble_kw = dict()
    elif not isinstance(ensemble_kw, dict):
        raise ValueError("tree_kw should be of type dict")

    if isinstance(sk_ensemble.estimators_, np.ndarray):
        ensemble_estimators = sk_ensemble.estimators_.ravel()
    elif isinstance(sk_ensemble.estimators_, Iterable):
        ensemble_estimators = sk_ensemble.estimators_

    ensemble_of_leaves = assemble_lucid_trees(
        ensemble_estimators,
        feature_names, print_precision, **tree_kw)

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

    try:  # retrieve loss function if there is one
        loss_function = sk_ensemble.loss_
    except AttributeError:  # if None default to mean_squared_error
        loss_function = mean_squared_error

    try:
        learning_rate = sk_ensemble.learning_rate
    except AttributeError:
        learning_rate = 1.0 / sk_ensemble.n_estimators

    return LucidSKEnsemble(
        ensemble_of_leaves, feature_names,
        init_estimator=init_estimator,
        loss_function=loss_function,
        learning_rate=learning_rate,
        **ensemble_kw)
