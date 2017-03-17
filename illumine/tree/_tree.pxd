"""
    Classes for storing tree structures

"""

import numpy as np
cimport numpy as cnp


cdef class TreeSplit:
    """ Representation of TreeSplit which contains
        feature_name, relation, threshold of a
        decision tree split
    """

    cdef int _feature
    cdef str _feature_name
    cdef str _relation
    cdef double _threshold
    cdef int _print_precision


cdef class TreeNode:
    """ Representation of each node in the decision tree.
        As opposed to TreeSplit, this contains pointers
        to the children. Also TreeNode does not contain the
        value nor relation.
    """

    cdef TreeNode parent
    cdef TreeNode left_child
    cdef TreeNode right_child
    cdef int _index
    cdef int _feature
    cdef double _threshold
    cdef double _value
    cdef bint _is_leaf


cdef class TreeStructure:
    """ Representation of TreeStructure which contains
        feature_name, relation, threshold of a
    """

    cdef TreeNode _root

    cpdef cnp.ndarray[double, ndim=1] decision_path(
             TreeStructure self,
             cnp.ndarray[double, ndim=2] X)

    cdef void _set_leaves(TreeStructure self, TreeNode tree_node,
                          double[:, :] X,
                          double[:] active_leaves,
                          int[:] indices)
