"""
    Cython module for the retrieve_leaf_path
    function used to build lucid-tree objects

    TODO:
        currently feature_names and
        print_precision are not being used
"""

cimport cython

import numpy as np
cimport numpy as cnp

from collections import Iterable

from .leaf cimport TreeSplit
from .leaf cimport TreeLeaf
from .leaftable import LeafTable


@cython.cdivision(True)
def build_leaf_tables(sk_trees, list feature_names, int print_precision=5):
    """ """
    cdef list tree_data
    tree_data = build_tree_leaves(
        *_accumulate_tree_attributes(sk_trees),
        # must be changed to strings to be passed into Cython function
        feature_names=list(map(str, feature_names)),
        print_precision=print_precision)

    cdef list tree_leaves
    cdef list leaftables = []

    for tree_leaves in tree_data:
        leaftables.append(
            LeafTable(tree_leaves))

    return leaftables


def _accumulate_tree_attributes(sk_trees):
    """ Gather the necessary data from Scikit-learn trees
        to construct TreeLeaf objects

    :param sk_trees: list of Scikit-learn decision trees
    """
    if not isinstance(sk_trees, Iterable):
        sk_trees = [sk_trees]

    cdef object sk_tree
    cdef list accm_values = []
    cdef list accm_node_samples = []
    cdef list accm_features = []
    cdef list accm_thresholds = []

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


@cython.cdivision(True)
cpdef list build_tree_leaves(list accm_values,
                             list accm_node_samples,
                             list accm_features,
                             list accm_thresholds,
                             list feature_names,
                             int print_precision):
    """ Deconstruct several trees using the data
        gathered by _accumulate_tree_attributes and
        build TreeLeaves out of them.
    """
    cdef list tree_data = []
    cdef tuple tup

    for tup in zip(accm_values,
                   accm_node_samples,
                   accm_features,
                   accm_thresholds):

        tree_data.append(_build_tree_leaves(
            values=tup[0],
            node_samples=tup[1],
            features=tup[2],
            thresholds=tup[3],
            feature_names=feature_names,
            print_precision=print_precision)
        )
    return tree_data


cdef list _build_tree_leaves(
    cnp.ndarray[double, ndim=3] values,
    cnp.ndarray[long, ndim=1] node_samples,
    cnp.ndarray[long, ndim=1] features,
    cnp.ndarray[double, ndim=1] thresholds,
    list feature_names,
    int print_precision):
    """ This function does most of the work in building TreeLeaf objects.
        Does a pre-order traversal over the tree attributes passed
        and creates TreeLeaf objects along the way.
    """
    cdef int n_splits = features.shape[0]
    if n_splits == 0:
        raise ValueError("The passed tree is empty!")

    cdef list tree_leaves = []
    cdef list tracker_stack = []        # used to track of child node's visited
    cdef list leaf_path = []
    cdef TreeSplit treesplit_tmp

    cdef int node_index
    for node_index in xrange(n_splits):
        if len(tracker_stack) != 0:
            # the last element of the tracker stack
            # refers to the parent of the current node
            tracker_stack[-1] += 1       # keep track # of children visited

        if features[node_index] != -2:      # visiting inner node
            tracker_stack.append(0)
            leaf_path.append(
                TreeSplit(
                    features[node_index],
                    thresholds[node_index],
                    feature_names[features[node_index]],
                    '<=')
            )

        else:  # visiting leaf/terminal node
            tree_leaves.append(
                TreeLeaf(leaf_path.copy(),
                         values[node_index][0][0],
                         node_samples[node_index])
            )

            if len(tracker_stack) > 0 and tracker_stack[-1] == 2:
                # pop out nodes that I am completely done with
                while(len(tracker_stack) > 0 and tracker_stack[-1] == 2):
                    leaf_path.pop()
                    tracker_stack.pop()

            if len(leaf_path) != 0:
                treesplit_tmp = leaf_path.pop()
                # switching to the right child of parent
                leaf_path.append(
                    TreeSplit(
                        treesplit_tmp.feature,
                        treesplit_tmp.threshold,
                        feature_names[treesplit_tmp.feature],
                        '>')
                )


    return tree_leaves
