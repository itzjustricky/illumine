"""
    Module containing functions to create
    LucidTree objects

"""

# from collections import Iterable
# from collections import OrderedDict
import numpy as np

# from .leaf_retrieval import deconstruct_trees
from .leaf_retrieval import build_leaf_tables

# from .lucid_tree import TreeLeaf
from .lucid_tree import LucidTree

__all__ = ['make_LucidTree']


def make_many_LucidTrees(sk_trees, feature_names, print_precision=5):
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

    :returns: LucidTree object indexed by their order in the
        pre-order traversal of the Decision Tree
    """
    lucid_trees = np.array(
        [LucidTree(leaf_table)
         for leaf_table in build_leaf_tables(sk_trees, list(feature_names), print_precision)],
        dtype=object)

    return lucid_trees


def make_LucidTree(sk_tree, feature_names, print_precision=5):
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

    :returns: LucidTree object indexed by their order in the
        pre-order traversal of the Decision Tree
    """
    return LucidTree(build_leaf_tables(
        sk_tree, list(feature_names), print_precision)[0])
