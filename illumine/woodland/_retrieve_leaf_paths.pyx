"""
    Cython module for the retrieve_leaf_path
    function used to build lucid-tree objects

"""

cimport cython
cimport numpy as np


@cython.cdivision(True)
def retrieve_tree_metas(accm_values,
                        accm_node_samples,
                        accm_features,
                        accm_thresholds,
                        np.ndarray feature_names,
                        int float_precision):
    tree_metas = []

    for tup in zip(accm_values,
                   accm_node_samples,
                   accm_features,
                   accm_thresholds):

        tree_metas.append(retrieve_leaf_path(
            values=tup[0],
            node_samples=tup[1],
            features=tup[2],
            thresholds=tup[3],
            feature_names=feature_names,
            float_precision=float_precision)
        )
    return tree_metas


cdef class TreeSplit:
    """ Representation of TreeSplit which contains
        feature_name, relation, threshold of a
        decision tree split
    """

    cdef str _feature_name
    cdef str _relation
    cdef double _threshold
    cdef int _float_precision

    def __cinit__(self,
                  str feature_name,
                  str relation,
                  double threshold,
                  int float_precision):
        self._feature_name = feature_name
        self._relation = relation
        self._threshold = threshold
        self._float_precision = float_precision

    @property
    def feature_name(self):
        return self._feature_name

    @property
    def relation(self):
        return self._relation

    @property
    def threshold(self):
        return self._threshold

    def __key(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return "{}{}{}".format(
            self.feature_name,
            self.relation,
            round(self.threshold, self._float_precision))

    def __repr__(self):
        return self.__str__()


cdef retrieve_leaf_path(np.ndarray[double, ndim=3] values,
                        np.ndarray[long, ndim=1] node_samples,
                        np.ndarray[long, ndim=1] features,
                        np.ndarray[double, ndim=1] thresholds,
                        np.ndarray feature_names,
                        int float_precision):
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
    n_splits = len(features)

    if n_splits == 0:
        raise ValueError("The passed tree is empty!")

    cdef list tree_meta = []
    cdef list tracker_stack = []  # a stack to track if all the children of a node is visited
    cdef list leaf_path = []      # ptr_stack keeps track of nodes

    cdef int node_index
    for node_index in xrange(n_splits):
        if len(tracker_stack) != 0:
            tracker_stack[-1] += 1  # visiting the child of the latest node

        if features[node_index] != -2:  # visiting inner node
            tracker_stack.append(0)
            tree_split = TreeSplit(
                feature_name=feature_names[features[node_index]],
                relation='<=',
                threshold=thresholds[node_index],
                float_precision=float_precision
            )
            leaf_path.append(tree_split)
            # tmp_split = leaf_path.pop()
            # print(tmp_split.relation)

        else:  # visiting leaf
            tree_meta.append((
                node_index,                            # leaf's index in pre-order traversal
                leaf_path.copy(),                      # path to the leaf
                float(round(values[node_index][0][0],  # leaf value
                            float_precision)),
                node_samples[node_index]               # number of samples at leaf
            ))

            if len(tracker_stack) > 0 and tracker_stack[-1] == 2:
                # pop out nodes that I am completely done with
                while(len(tracker_stack) > 0 and tracker_stack[-1] == 2):
                    leaf_path.pop()
                    tracker_stack.pop()
            if len(leaf_path) != 0:
                tmp_split = leaf_path.pop()
                new_split = TreeSplit(
                    feature_name=tmp_split.feature_name,
                    relation='>',
                    threshold=tmp_split.threshold,
                    float_precision=float_precision
                )
                leaf_path.append(new_split)

    return tree_meta
