"""
    Module of Leaf objects which will store the
    data needed to decide the decision path and
    value at the terminal nodes.

"""

import numpy as np
cimport numpy as cnp

# ctypedef cnp.npy_float32 DTYPE_t          # Type of X
# ctypedef cnp.npy_intp SIZE_t              # Type for indices and counters
ctypedef cnp.npy_float64 DOUBLE_t           # Type of y, sample_weight
ctypedef cnp.npy_int32 INT32_t              # Signed 32 bit integer
ctypedef cnp.npy_uint32 UINT32_t            # Unsigned 32 bit integer
# ctypedef unordered_map[int, vector[int]] int_vint_map

ctypedef bint (*relation_fn)(double value, double threshold) nogil


cdef class TreeSplit:
    cdef readonly INT32_t feature
    cdef readonly DOUBLE_t threshold
    cdef readonly str feature_name
    cdef readonly str relation
    cdef relation_fn relate

    # methods of TreeSplit class
    cdef bint apply(self, double value) nogil


cdef class TreeLeaf:
    cdef readonly double value
    cdef readonly UINT32_t n_samples
    cdef public dict f_to_split_map
    cdef public list tree_splits

    # methods of TreeLeaf class
    cdef void apply(self, double[:, :] X, int[:] b)
    cdef void _apply_to_feature(self, int feature, int[:] split_inds,
                                double[:, :] X, int[:] b_vector)
    cdef void _dense_apply(self, double[:, :] X, int[:] b_vector)
