"""
    LeafTable is an object that
    stores a list of TreeLeaf

    TODO: currently everything is quite
        stupid, will be upgraded after lower levels
        are well-tested

"""

cimport cython

import numpy as np
cimport numpy as cnp

from .leaf cimport TreeLeaf


cdef class LeafTable:

    def __cinit__(self, list tree_leaves):
        self.tree_leaves = tree_leaves
        self.n_leaves = len(tree_leaves)

    cdef void _apply(self, double[:, :] X, int[:, :] B):
        """ Compute an activation matrix with shape NxL
            where N is the # of samples
                  L is the # of leaves in the LeafTable
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]

        cdef int[:] b_tmp
        cdef TreeLeaf tree_leaf

        cdef int i, l_iter
        for l_iter in range(self.n_leaves):
            b_tmp = np.ones(X.shape[0], dtype=np.int32)

            tree_leaf = self.tree_leaves[l_iter]
            tree_leaf.apply(X, b_tmp)
            for i in range(n_samples):
                B[i, l_iter] = b_tmp[i]

    @cython.cdivision(True)
    cpdef cnp.ndarray apply(self, cnp.ndarray[double, ndim=2] X):
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[int, ndim=2] B = \
            np.zeros((n_samples, self.n_leaves), dtype=np.int32)
        self._apply(X, B)

        return B

    cdef void _predict(self, double[:, :] X, double[:] y_pred):
        cdef int n_samples = X.shape[0]

        # some temporary vars to store state of leaf in loop
        cdef int[:] b_tmp           # store bitmap of rows activated
        cdef double leaf_value      # store value of leaf
        cdef TreeLeaf tree_leaf     # store leaf extension type

        cdef int i, l_iter
        for l_iter in range(self.n_leaves):
            tree_leaf = self.tree_leaves[l_iter]
            leaf_value = tree_leaf.value
            # compute the activated rows by leaf
            b_tmp = np.ones(X.shape[0], dtype=np.int32)
            tree_leaf.apply(X, b_tmp)

            for i in range(n_samples):
                if b_tmp[i]:
                    y_pred[i] += leaf_value

    @cython.cdivision(True)
    cpdef cnp.ndarray[double, ndim=1] predict(self, cnp.ndarray[double, ndim=2] X):
        """ Provide a prediction for feature matrix X """
        cdef int n_samples = X.shape[0]

        cdef cnp.ndarray[double, ndim=1] y_pred = \
            np.zeros(n_samples, dtype=np.float64)
        self._predict(X, y_pred)
        return y_pred

    def __len__(self):
        return self.n_leaves
