"""
    Module of Leaf objects which will store the
    data needed to decide the decision path and
    value at the terminal nodes.

"""

import numpy as np
cimport numpy as cnp

# ctypedef cnp.npy_float32 DTYPE_t         # Type of X
# ctypedef cnp.npy_float64 DOUBLE_t        # Type of y, sample_weight
# ctypedef cnp.npy_intp SIZE_t             # Type for indices and counters
ctypedef cnp.npy_int32 INT32_t           # Signed 32 bit integer
ctypedef cnp.npy_uint32 UINT32_t          # Unsigned 32 bit integer


cdef class TreeSplit:
    cdef readonly INT32_t feature
    cdef readonly str feature_name
    cdef readonly str relation
    cdef readonly double threshold
    cdef readonly UINT32_t print_precision

    cdef bint apply(self, double value)


cdef class DecisionPath:
    cdef readonly str key
    cdef readonly int[:] relevant_features
    cdef public dict feature_map

    cdef bint apply(self, double value, int feature)


cdef class TreeLeaf:
    """ Object representation a leaf (terminal node) of a decision tree.
        Stores the path to the leaf and the value associated with the leaf.
    """

    cdef readonly double value
    cdef readonly UINT32_t n_samples
    cdef readonly int[:] relevant_features
    cdef public DecisionPath decision_path
    cdef readonly str _cached_repr

    cdef int[:] _dense_apply(self, double[:, :] X, int feature)
