"""
    Cython module for the retrieve_leaf_paths
    function used to build lucid-tree objects

"""

cimport cython
cimport numpy as np


@cython.cdivision(False)
def retrieve_leaf_paths(np.ndarray[double, mode="c", ndim=3] values,
                        np.ndarray node_samples,
                        np.ndarray features,
                        np.ndarray thresholds,
                        np.ndarray children_right,
                        np.ndarray feature_names,
                        int float_precision):
    """ Gather all the leaves and keep track of the paths that
        define the leaf.
    """
    n_splits = len(features)
    leaf_meta = []

    tracker_stack = []  # a stack to track if all the children of a node is visited
    leaf_path = []      # ptr_stack keeps track of nodes
    for node_index in range(n_splits):
        if len(tracker_stack) != 0:
            tracker_stack[-1] += 1  # visiting the child of the latest node

        if features[node_index] != -2:  # visiting inner node
            tracker_stack.append(0)
            append_str = "{}<={}".format(
                feature_names[features[node_index]],
                float(round(thresholds[node_index], float_precision)))
            leaf_path.append(append_str)

        else:  # visiting leaf
            leaf_meta.append((
                node_index,                            # leaf index in pre-order traversal
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

    return leaf_meta
