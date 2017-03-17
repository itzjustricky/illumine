"""
    Classes for storing tree structures

"""

cimport cython

import numpy as np
cimport numpy as cnp


cdef class TreeSplit:
    """ Representation of TreeSplit which contains
        feature_name, relation, threshold of a
        decision tree split
    """

    def __cinit__(self,
                  int feature,
                  str feature_name,
                  str relation,
                  double threshold,
                  int print_precision):
        self._feature = feature
        self._feature_name = feature_name
        self._relation = relation
        self._threshold = threshold
        self._print_precision = print_precision

    @property
    def feature(self):
        return self._feature

    @property
    def feature_name(self):
        return self._feature_name

    @property
    def relation(self):
        return self._relation

    @property
    def threshold(self):
        return self._threshold

    @property
    def print_precision(self):
        return self._print_precision

    def __str__(self):
        return "TS({}{}{})".format(
            self.feature_name,
            self.relation,
            round(self.threshold, self._print_precision))

    def __reduce__(self):
        return (self.__class__, (
            self._feature,
            self._feature_name,
            self._relation,
            self._threshold,
            self._print_precision)
        )

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __richcmp__(x, y, int op):
        if op == 0:
            return str(x) < str(y)
        if op == 2:
            return str(x) == str(y)
        if op == 4:
            return str(x) > str(y)
        if op == 1:
            return str(x) <= str(y)
        if op == 3:
            return str(x) != str(y)
        if op == 5:
            return str(x) >= str(y)


cdef class TreeNode:
    """ Representation of each node in the decision tree.
        As opposed to TreeSplit, this contains pointers
        to the children. Also TreeNode does not contain the
        value nor relation.
    """

    def __cinit__(self, int index=0, int feature=0,
                  double threshold=0.0, double value=0.0):
        self._index = index      # pre-order traversal index
        self._feature = feature
        self._threshold = threshold
        self._value = value

    def set_left_child(self, TreeNode left_child):
        self.left_child = left_child
        left_child.parent = self

    def set_right_child(self, TreeNode right_child):
        self.right_child = right_child
        right_child.parent = self

    @property
    def index(self):
        return self._index

    @property
    def feature(self):
        return self._feature

    @property
    def threshold(self):
        return self._threshold

    @property
    def value(self):
        return self._value

    @property
    def is_leaf(self):
        return self._is_leaf

    @is_leaf.setter
    def is_leaf(self, leaf_flag):
        if not isinstance(leaf_flag, bool):
            raise ValueError("Argument leaf_flag should be of type boolean")
        self._is_leaf = leaf_flag

    # def __reduce__(self):
    #     return (self.__class__, (
    #         self._index,
    #         self._feature,
    #         self._threshold,
    #         self._value)
    #     )

    def __getstate__(self):
        return (
            self._index,
            self._feature,
            self._threshold,
            self._value,
            self._is_leaf,
            self.parent,
            self.left_child,
            self.right_child,
        )

    def __setstate__(self, state_tuple):
        # print("Loading from dictionary: {}".format(attr_dict))
        self._index, self._feature, self._threshold, \
        self._value, self._is_leaf, self.parent, \
        self.left_child, self.right_child = state_tuple

        # self._index = attr_dict['_index']
        # self._feature = attr_dict['_feature']
        # self._threshold = attr_dict['_threshold']
        # self._value = attr_dict['_value']
        # self._is_leaf = attr_dict['_is_leaf']
        # self.parent = attr_dict['parent']
        # self.left_child = attr_dict['left_child']
        # self.right_child = attr_dict['right_child']


cdef class TreeStructure:
    """ Representation of TreeStructure which contains
        feature_name, relation, threshold of a
    """

    def __cinit__(self, TreeNode root):
        self._root = root

    @cython.cdivision(True)
    cpdef cnp.ndarray[double, ndim=1] decision_path(
        TreeStructure self,
        cnp.ndarray[double, ndim=2] X):
        """ Find the leafs activated for each
            row in the 2d input matrix X
        """
        cdef cnp.ndarray[int, ndim=1] indices = np.arange(X.shape[0], dtype=np.int32)
        cdef cnp.ndarray[double, ndim=1] active_leaves = \
            np.zeros(X.shape[0], dtype=np.float64)
        self._set_leaves(self._root, X, active_leaves, indices)

        return active_leaves

    cdef void _set_leaves(TreeStructure self, TreeNode tree_node,
                          double[:, :] X,
                          double[:] active_leaves,
                          int[:] indices):
                          # cnp.ndarray[double, ndim=2] X,
                          # cnp.ndarray[double, ndim=1] active_leaves,
                          # cnp.ndarray[int, ndim=1] indices):
        """ Recursively set the indices for the active_leaves 1darray """
        cdef int i, n_rows
        n_rows = X.shape[0]

        cdef int node_feature = tree_node.feature
        cdef double node_value = tree_node.value
        cdef double node_threshold = tree_node.threshold
        # cdef cnp.ndarray[int, ndim=1] tmp_left_inds, tmp_right_inds
        cdef int[:] left_inds, right_inds

        if tree_node.is_leaf:
            with nogil:
                for i in range(n_rows):
                    if indices[i] != -1:
                        active_leaves[i] = node_value
        else:
            left_inds = np.full(n_rows, -1, dtype=np.int32)
            right_inds = np.full(n_rows, -1, dtype=np.int32)

            with nogil:
                for i in range(n_rows):
                    if indices[i] != -1:
                        if X[i, node_feature] <= node_threshold:
                            left_inds[i] = i
                        else:
                            right_inds[i] = i

            self._set_leaves(tree_node.left_child, X, active_leaves, left_inds)
            self._set_leaves(tree_node.right_child, X, active_leaves, right_inds)


    def __reduce__(self):
        return (self.__class__, tuple((self._root,)))
