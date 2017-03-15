"""
    Cython module for the retrieve_leaf_path
    function used to build lucid-tree objects

"""

cimport cython

cimport numpy as cnp

from . cimport _tree
from . import _tree


@cython.cdivision(True)
def deconstruct_trees(list accm_values,
                      list accm_node_samples,
                      list accm_features,
                      list accm_thresholds,
                      list feature_names,
                      int print_precision):
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


cdef tuple _deconstruct_tree(cnp.ndarray[double, ndim=3] values,
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
    cdef int n_splits
    n_splits = len(features)

    if n_splits == 0:
        raise ValueError("The passed tree is empty!")

    cdef list tree_meta = []
    # tracker_stack latest value will be
    cdef list tracker_stack = []  # a stack to track if all children of node is visited
    cdef list leaf_path = []      # ptr_stack keeps track of nodes

    # Set up variables for storing tree structure
    cdef _tree.TreeNode root_node, node_ptr
    cdef _tree.TreeStructure tree_struct
    root_node = _tree.TreeNode(0, features[0], thresholds[0], 0.0)
    tree_struct = _tree.TreeStructure(root_node)
    node_ptr = root_node

    cdef int node_index
    for node_index in xrange(n_splits):
        if len(tracker_stack) != 0:
            # the last element of the tracker stack
            # refers to the parent of the current node
            tracker_stack[-1] += 1  # keep track # of children visited

            # build tree structure here
            if tracker_stack[-1] == 1:      # visiting left node of parent
                node_ptr.set_left_child(
                        _tree.TreeNode(node_index, features[node_index],
                                       thresholds[node_index], values[node_index]))
                node_ptr = node_ptr.left_child
            elif tracker_stack[-1] == 2:    # visiting right node of parent
                node_ptr.set_right_child(
                        _tree.TreeNode(node_index, features[node_index],
                                       thresholds[node_index], values[node_index]))
                node_ptr = node_ptr.right_child

        if features[node_index] != -2:  # visiting inner node
            tracker_stack.append(0)
            tree_split = _tree.TreeSplit(
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
            node_ptr.is_leaf = True

            if len(tracker_stack) > 0 and tracker_stack[-1] == 2:
                # pop out nodes that I am completely done with
                while(len(tracker_stack) > 0 and tracker_stack[-1] == 2):
                    leaf_path.pop()
                    tracker_stack.pop()
                    node_ptr = node_ptr.parent

            if len(leaf_path) != 0:
                tmp_split = leaf_path.pop()
                new_split = _tree.TreeSplit(
                    feature=tmp_split.feature,
                    feature_name=tmp_split.feature_name,
                    relation='>',
                    threshold=tmp_split.threshold,
                    print_precision=print_precision
                )
                leaf_path.append(new_split)
                node_ptr = node_ptr.parent

    return tree_struct, tree_meta
