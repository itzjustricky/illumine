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

ctypedef bint (*relation_fn)(double value, double threshold) nogil


cdef class TreeSplit:
    cdef readonly INT32_t feature               # col index of feature in X
    cdef readonly DOUBLE_t threshold            # threshold value for split
    cdef readonly INT32_t print_precision       # decimals to print for repr
    cdef readonly str float_formatter           # uses print_precision for formatting floats
    cdef readonly str feature_name              # name of the feature
    cdef readonly str relation                  # the relation i.e. '<=' or '>'
    cdef relation_fn relate                     # ptr to relation function

    # methods of TreeSplit class
    cdef bint apply(self, double value) nogil


cdef class TreeLeaf:
    cdef public DOUBLE_t value          # value of the leaf/terminal-node
    cdef public list tree_splits        # TreeSplits that make up the TreeLeaf
    cdef public dict f_to_split_map     # map from features to splits
    cdef readonly INT32_t leaf_hash     # hash code of a set of TreeSplit

    # methods of TreeLeaf class
    cdef void apply(self, double[:, :] X, unsigned char[:] b_vector)
    cdef void _apply_to_feature(
        self, int feature, int[:] split_inds,
        double[:, :] X, unsigned char[:] b_vector)
    cdef void _dense_apply(self, double[:, :] X, unsigned char[:] b_vector)
