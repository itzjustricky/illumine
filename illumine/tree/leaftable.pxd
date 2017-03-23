"""
    LeafTable is an object that keeps a map
    of relevant leaves per feature.

    This will be the class that handles prediction
    and activation matrix computation.

"""

import numpy as np
cimport numpy as cnp

from cpython cimport PyObject

# from .leaf cimport TreeLeaf

# ctypedef cnp.npy_float32 DTYPE_t          # Type of X
# ctypedef cnp.npy_intp SIZE_t              # Type for indices and counters
ctypedef cnp.npy_float64 DOUBLE_t           # Type of y, sample_weight
ctypedef cnp.npy_int32 INT32_t              # Signed 32 bit integer
ctypedef cnp.npy_uint32 UINT32_t            # Unsigned 32 bit integer


cdef class LeafTable:
    cdef readonly int n_leaves
    cdef public list tree_leaves

    cdef cnp.ndarray _dense_apply(self, double[:, :] X)
    cdef object _sparse_apply(self, double[:, :] X)
    cdef cnp.ndarray _dense_predict(self, double[:, :] X)

    cpdef object apply(self, cnp.ndarray[double, ndim=2] X, bint sparse=?)
    cpdef cnp.ndarray predict(self, cnp.ndarray[double, ndim=2] X)
