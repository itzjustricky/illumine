"""
    Description:
        Methods for analyzing tree nodes following the Scikit-Learn API

     make_LucidSKTree : factory method for LucidSKTree;
     make_LucidSKEnsemble : factory method for LucidSKEnsemble;
     unique_leaves_per_sample : function
     get_tree_predictions : function

    TODO:
        * if in the future, experience make_LucidSKEnsemble bottleneck use
            multiprocessing library to unravel trees in parallel
        * separate out leaf path retrieval algorithm from make_LucidSKTree
        * add an extra parameter to choose which leaves to consider

    @author: Ricky
"""

from collections import OrderedDict
from collections import Iterable

import numpy as np

from .leaf_objects import SKTreeNode
from .leaf_objects import LucidSKTree
from .leaf_objects import LucidSKEnsemble

from ._retrieve_leaf_paths import retrieve_tree_metas

__all__ = ['make_LucidSKTree', 'make_LucidSKEnsemble']


def assemble_lucid_trees(sk_trees, feature_names, float_precision, **tree_kw):
    """ This function is used to gather the paths to all the
        leaves given some of the Scikit-learn attributes
        of a decision tree.
    """
    tree_metas = retrieve_tree_metas(
        *_accumulate_tree_attributes(sk_trees),
        feature_names=np.array(feature_names),
        float_precision=float_precision)

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
    accm_children_right = []

    for sk_tree in sk_trees:
        accm_values.append(sk_tree.tree_.value)
        accm_node_samples.append(sk_tree.tree_.n_node_samples)
        accm_features.append(sk_tree.tree_.feature)
        accm_thresholds.append(sk_tree.tree_.threshold)
        accm_children_right.append(sk_tree.tree_.children_right)
    return (
        accm_values,
        accm_node_samples,
        accm_features,
        accm_thresholds,
        accm_children_right
    )


def make_LucidSKTree(sk_tree, feature_names, float_precision=5,
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
    :param float_precision (int): to determine what number the node
        values, thresholds are rounded to
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
        sk_tree, feature_names, float_precision, **tree_kw)[0]


def make_LucidSKEnsemble(sk_ensemble, feature_names, float_precision=5,
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
    :param float_precision (int): to determine what number the node
        values, thresholds are rounded to
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

    ensemble_of_leaves = assemble_lucid_trees(
        sk_ensemble.estimators_.ravel(),
        feature_names, float_precision, **tree_kw)

    if init_estimator is None:
        init_estimator = sk_ensemble._init_decision_function
    elif not callable(init_estimator):
        raise ValueError(
            "The init_estimator should be a callable function that "
            "takes X (feature matrix) as an argument.")

    return LucidSKEnsemble(
        ensemble_of_leaves, feature_names,
        init_estimator=init_estimator,
        learning_rate=sk_ensemble.learning_rate,
        **ensemble_kw)
