"""
    Cython module for the retrieve_leaf_paths
    function used to build lucid-tree objects

"""

cimport cython
cimport numpy as np


@cython.cdivision(True)
def retrieve_tree_metas(accm_values,
                        accm_node_samples,
                        accm_features,
                        accm_thresholds,
                        accm_children_right,
                        np.ndarray feature_names,
                        int float_precision):
    tree_metas = []

    for tup in zip(accm_values,
                   accm_node_samples,
                   accm_features,
                   accm_thresholds,
                   accm_children_right):

        tree_metas.append(retrieve_leaf_paths(
            values=tup[0],
            node_samples=tup[1],
            features=tup[2],
            thresholds=tup[3],
            children_right=tup[4],
            feature_names=feature_names,
            float_precision=float_precision)
        )
    return tree_metas


cdef retrieve_leaf_paths(np.ndarray[double, ndim=3] values,
                         np.ndarray[long, ndim=1] node_samples,
                         np.ndarray[long, ndim=1] features,
                         np.ndarray[double, ndim=1] thresholds,
                         np.ndarray[long, ndim=1] children_right,
                         np.ndarray feature_names,
                         int float_precision):
    """ Gather all the leaves of a tree and keep track of the
        paths that define the leaf.
    """
    n_splits = len(features)

    if n_splits == 0:
        raise ValueError("The passed tree is empty!")

    tree_meta = []
    tracker_stack = []  # a stack to track if all the children of a node is visited
    leaf_path = []      # ptr_stack keeps track of nodes

    cdef int node_index
    for node_index in xrange(n_splits):
        if len(tracker_stack) != 0:
            tracker_stack[-1] += 1  # visiting the child of the latest node

        if features[node_index] != -2:  # visiting inner node
            tracker_stack.append(0)
            append_str = "{}<={}".format(
                feature_names[features[node_index]],
                float(round(thresholds[node_index], float_precision)))
            leaf_path.append(append_str)

        else:  # visiting leaf
            tree_meta.append((
                node_index,                            # leaf's index in pre-order traversal
                leaf_path.copy(),                      # path to the leaf
                float(round(values[node_index][0][0],  # leaf value
                            float_precision)),
                node_samples[node_index]               # number of samples at leaf
            ))

            if node_index in children_right:
                # pop out nodes that I am completely done with
                while(len(tracker_stack) > 0 and tracker_stack[-1] == 2):
                    leaf_path.pop()
                    tracker_stack.pop()
            if len(leaf_path) != 0:
                leaf_path[-1] = leaf_path[-1].replace("<=", ">")

    return tree_meta
