"""
    LeafTable is an object that stores a list of TreeLeaf

"""

cimport cython

import numpy as np
cimport numpy as cnp
# from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

from .leaf cimport TreeLeaf


cdef class LeafTable:

    def __cinit__(self, list tree_leaves):
        self.tree_leaves = tree_leaves
        self.n_leaves = len(tree_leaves)

    cdef cnp.ndarray _dense_apply(self, double[:, :] X):
        """ Compute the activations per leaf for the passed feature
            matrix X and store the results in view B

            where N is the # of samples
                  L is the # of leaves in the LeafTable
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]

        cdef cnp.ndarray[cnp.uint8_t, ndim=2] B = \
            np.zeros((n_samples, self.n_leaves), dtype=np.uint8)

        cdef unsigned char[:, :] B_view = B
        cdef unsigned char[:] b_tmp
        cdef TreeLeaf tree_leaf

        cdef int i, l_iter
        for l_iter in range(self.n_leaves):
            b_tmp = np.ones(X.shape[0], dtype=np.uint8)

            tree_leaf = self.tree_leaves[l_iter]
            tree_leaf.apply(X, b_tmp)
            with nogil:
                for i in range(n_samples):
                    B_view[i, l_iter] = b_tmp[i]

        return B

    cdef object _sparse_apply(self, double[:, :] X):
        """ Compute the activations per leaf for the passed feature
            matrix X and store the results in view B

            where N is the # of samples
                  L is the # of leaves in the LeafTable
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]

        cdef unsigned char[:] b_tmp
        cdef TreeLeaf tree_leaf
        cdef list row_inds = []
        cdef list col_inds = []

        cdef int i, l_iter
        for l_iter in range(self.n_leaves):
            b_tmp = np.ones(X.shape[0], dtype=np.uint8)

            tree_leaf = self.tree_leaves[l_iter]
            tree_leaf.apply(X, b_tmp)
            for i in range(n_samples):
                if b_tmp[i]:
                    row_inds.append(i)
                    col_inds.append(l_iter)

        return csr_matrix(
            (np.ones(len(row_inds), dtype=np.bool), (row_inds, col_inds)),
            shape=(n_samples, self.n_leaves))

    cdef cnp.ndarray _dense_predict(self, double[:, :] X):
        """ Create the predictions from the leaves in tree_leaves
            and store them in the view y_pred
        """
        cdef int n_samples = X.shape[0]
        cdef cnp.ndarray[double, ndim=1] y_pred = \
            np.zeros(n_samples, dtype=np.float64)
        cdef double[:] y_pred_view = y_pred

        # some temporary vars to store state of leaf in loop
        cdef unsigned char[:] b_tmp     # store bitmap of rows activated
        cdef double leaf_value          # store value of leaf
        cdef TreeLeaf tree_leaf         # store leaf extension type

        cdef int i, l_iter
        for l_iter in range(self.n_leaves):
            tree_leaf = self.tree_leaves[l_iter]
            leaf_value = tree_leaf.value
            # compute the activated rows by leaf
            b_tmp = np.ones(X.shape[0], dtype=np.uint8)
            tree_leaf.apply(X, b_tmp)
            with nogil:
                for i in range(n_samples):
                    if b_tmp[i]:
                        y_pred_view[i] += leaf_value
        return y_pred

    @cython.cdivision(True)
    cpdef object apply(self, cnp.ndarray[double, ndim=2] X, bint sparse=False):
        """ Compute the activations per leaf for the passed features """

        if not sparse:
            return self._dense_apply(X)
        else:
            return self._sparse_apply(X)


    @cython.cdivision(True)
    cpdef cnp.ndarray predict(self, cnp.ndarray[double, ndim=2] X):
        """ Provide a prediction for feature matrix X """
        return self._dense_predict(X)

    def __len__(self):
        return self.n_leaves

    def __iter__(self):
        return iter(self.tree_leaves)
