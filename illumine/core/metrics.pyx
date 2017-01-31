
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
                         np.ndarray[double, ndim=1] y_pred):
    cdef np.ndarray[double, ndim=1] err
    err = y_true - y_pred

    return -(err * err).sum()  / y_true.shape[0]


# Returns negative mean-absolute deviation
cdef double negative_mad(np.ndarray[double, ndim=1] y_true,
                         np.ndarray[double, ndim=1] y_pred):
    cdef np.ndarray[double, ndim=1] err
    err = y_true - y_pred

    cdef double absolute_err_sum = 0.0
    cdef double ele
    for ele in err:
        absolute_err_sum += abs(ele)

    return -absolute_err_sum / y_true.shape[0]


# Returns the R-squared measure
cdef double rsquared(np.ndarray[double, ndim=1] y_true,
                     np.ndarray[double, ndim=1] y_pred):

    return 1.0 + negative_mse(y_true, y_pred) / y_true.var()


# Returns the % of signs matched between two arrays
cdef double sign_match(np.ndarray[double, ndim=1] y_true,
                       np.ndarray[double, ndim=1] y_pred):
    cdef np.ndarray[double, ndim=1] sign_match
    sign_match = (y_true.sign() == y_pred.sign())
    return sign_match.sum() / y_true.shape[0]
