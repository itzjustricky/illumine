"""
    LeafTable is an object that keeps a map
    of relevant leaves per .

"""

cimport cython

import numpy as np
cimport numpy as cnp


cdef class LeafTable:

    def __cinit__(self, list tree_leaves):
        self.tree_leaves = tree_leaves
        self.n_leaves = len(tree_leaves)

    @cython.cdivision(True)
    cpdef cnp.ndarray apply(self, cnp.ndarray[double, ndim=2] X):
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[bint, ndim=2] B = \
            np.zeros((n_samples, self.n_leaves), dtype=bool)
        self._apply(X, B)

        return B

    cdef void _apply(self, double[:, :] X, bint[:, :] B):
        """ Compute an activation matrix with shape NxL
            where N is the # of samples
                  L is the # of leaves in the LeafTable
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        # cdef bint[:, :] B = np.zeros((n_samples, self.n_leaves), dtype=bool)
        cdef bint[:] bcol_tmp

        cdef int i, l_iter
        for l_iter in range(self.n_leaves):
            bcol_tmp = self.tree_leaves[l_iter].apply(X)
            for i in range(n_samples):
                B[i, l_iter] = bcol_tmp[i]

    @cython.cdivision(True)
    cpdef cnp.ndarray[double, ndim=1] predict(self, cnp.ndarray[double, ndim=2] X):
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[double, ndim=1] y_pred = np.zeros(n_samples, dtype=np.float64)
        self._predict(X, y_pred)

        return y_pred

    cdef void _predict(self, double[:, :] X, double[:] y_pred):
        cdef int n_samples = X.shape[0]

        cdef bint[:] bcol_tmp
        cdef double leaf_value
        # cdef double[:] y_pred = np.zeros(n_samples, dtype=np.float64)

        cdef int i, l_iter
        for l_iter in range(self.n_leaves):
            leaf_value = self.tree_leaves[l_iter].value
            bcol_tmp = self.tree_leaves[l_iter].apply(X)

            for i in range(n_samples):
                if bcol_tmp[i]:
                    y_pred[i] += leaf_value

    def __len__(self):
        return self.n_leaves
