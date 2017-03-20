"""
    Cython module for the retrieve_leaf_path
    function used to build lucid-tree objects

"""

cimport cython

import numpy as np
cimport numpy as cnp
from collections import Iterable

from .leaf cimport TreeSplit
# from .leaf cimport DecisionPath
from .leaf cimport TreeLeaf
from .leaftable import LeafTable


cpdef _accumulate_tree_attributes(sk_trees):
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
def construct_leaf_tables(sk_trees, list feature_names, int print_precision=5):
    cdef list tree_data
    tree_data = deconstruct_trees(
        *_accumulate_tree_attributes(sk_trees),
        # must be changed to strings to be passed into Cython function
        feature_names=list(map(str, feature_names)),
        print_precision=print_precision)

    cdef int leaf_ind
    cdef list tree_datum, tree_leaves
    cdef list leaftables = []
    for tree_datum in tree_data:
        tree_leaves = []

        for leaf_meta in tree_datum:
            leaf_ind = leaf_meta[0]
            tree_leaves.append(TreeLeaf(*leaf_meta[1:]))

        leaftables.append(LeafTable(tree_leaves))

    return leaftables


@cython.cdivision(True)
def deconstruct_trees(list accm_values,
                      list accm_node_samples,
                      list accm_features,
                      list accm_thresholds,
                      list feature_names,
                      int print_precision):
    """ Deconstruct several trees using the data
        gathered by _accumulate_tree_attributes
    """
    cdef list tree_data = []
    cdef tuple tup

    for tup in zip(accm_values,
                   accm_node_samples,
                   accm_features,
                   accm_thresholds):

        tree_data.append(_deconstruct_tree(
            values=tup[0],
            node_samples=tup[1],
            features=tup[2],
            thresholds=tup[3],
            feature_names=feature_names,
            print_precision=print_precision)
        )
    return tree_data


cdef list _deconstruct_tree(cnp.ndarray[double, ndim=3] values,
                             cnp.ndarray[long, ndim=1] node_samples,
                             cnp.ndarray[long, ndim=1] features,
                             cnp.ndarray[double, ndim=1] thresholds,
                             list feature_names,
                             int print_precision):
    """ Gather all the leaves of a tree and keep track of the
        paths that define the leaf.

    :returns (list): tree_meta a list of meta-data of a leaf node
        of a tree. The meta-data is as ordered:
        index - index of leaf in pre-order traversal of tree
        tree_split - meta-data of the leaf-path of tree consists of
            * feature_name
            * relation
            * threshold
        node_samples - # of samples that enter the leaf node from
            the training data
    """
    cdef int n_splits = len(features)

    if n_splits == 0:
        raise ValueError("The passed tree is empty!")

    cdef list tree_meta = []
    cdef list tracker_stack = []  # a stack to track if all children of node is visited
    cdef list leaf_path = []      # ptr_stack keeps track of nodes

    cdef int node_index
    for node_index in xrange(n_splits):
        if len(tracker_stack) != 0:
            # the last element of the tracker stack
            # refers to the parent of the current node
            tracker_stack[-1] += 1  # keep track # of children visited

        if features[node_index] != -2:  # visiting inner node
            tracker_stack.append(0)
            tree_split = TreeSplit(
                feature=features[node_index],
                feature_name=feature_names[features[node_index]],
                relation='<=',
                threshold=thresholds[node_index],
                print_precision=print_precision
            )
            leaf_path.append(tree_split)

        else:  # visiting leaf/terminal node
            tree_meta.append((
                node_index,                 # leaf's index in pre-order traversal
                leaf_path.copy(),           # path to the leaf
                values[node_index][0][0],   # leaf value
                node_samples[node_index]    # number of samples at leaf
            ))

            if len(tracker_stack) > 0 and tracker_stack[-1] == 2:
                # pop out nodes that I am completely done with
                while(len(tracker_stack) > 0 and tracker_stack[-1] == 2):
                    leaf_path.pop()
                    tracker_stack.pop()

            if len(leaf_path) != 0:
                tmp_split = leaf_path.pop()
                new_split = TreeSplit(
                    feature=tmp_split.feature,
                    feature_name=tmp_split.feature_name,
                    relation='>',
                    threshold=tmp_split.threshold,
                    print_precision=print_precision
                )
                leaf_path.append(new_split)

    return tree_meta
