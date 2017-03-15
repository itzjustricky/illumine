"""
    Module containing functions to create
    LucidTree objects

"""

from collections import Iterable
# from collections import OrderedDict

from .leaf_retrieval import deconstruct_trees

from .lucid_tree import TreeLeaf
from .lucid_tree import LucidTree

__all__ = ['make_LucidTree']


def assemble_lucid_trees(sk_trees, feature_names, print_precision, **tree_kws):
    """ This function is used to gather the paths to all the
        leaves given some of the Scikit-learn attributes
        of a decision tree.
    """
    tree_data = deconstruct_trees(
        *_accumulate_tree_attributes(sk_trees),
        # must be changed to strings to be passed into Cython function
        feature_names=list(map(str, feature_names)),
        print_precision=print_precision)

    lucid_trees = []
    for tree_datum in tree_data:
        tree_leaves = dict()

        tree_struct, leaf_metas = tree_datum
        for leaf_meta in leaf_metas:
            leaf_ind = leaf_meta[0]
            tree_leaves[leaf_ind] = TreeLeaf(*leaf_meta[1:])

        lucid_trees.append(
            LucidTree(tree_struct, tree_leaves, feature_names, **tree_kws))

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


def make_LucidTree(sk_tree, feature_names, print_precision=5, tree_kws=None):
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
    :param tree_kws (dict): key-word arguments to be passed into
        LucidTree's constructor

    :returns: LucidTree object indexed by their order in the
        pre-order traversal of the Decision Tree
    """
    if tree_kws is None:
        tree_kws = dict()
    elif not isinstance(tree_kws, dict):
        raise ValueError("tree_kws should be of type dict")
    return assemble_lucid_trees(
        sk_tree, feature_names, print_precision, **tree_kws)[0]
