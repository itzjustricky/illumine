"""
    Description:
        Cython metric functions so they
        can be called in Cython modules.

        Created for optimization purposes.

    @author: Ricky
"""

cimport cython
cimport numpy as np


# Returns negative mean-squared error
cdef double negative_mse(np.ndarray[double, ndim=1] y_true,
                         np.ndarray[double, ndim=1] y_pred)


# Returns negative least-absolute deviation
cdef double negative_mad(np.ndarray[double, ndim=1] y_true,
                         np.ndarray[double, ndim=1] y_pred)


# Returns the R-squared measure
cdef double rsquared(np.ndarray[double, ndim=1] y_true,
                     np.ndarray[double, ndim=1] y_pred)


# Returns the % of signs matched between two arrays
cdef double sign_match(np.ndarray[double, ndim=1] y_true,
                       np.ndarray[double, ndim=1] y_pred)
