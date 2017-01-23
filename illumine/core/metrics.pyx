
"""
    Description:
        Cython metric functions so they
        can be called in Cython modules.

        Created for optimization purposes.

    @author: Ricky
"""

cimport cython
cimport numpy as np


cdef double _mean(np.ndarray[double, ndim=1] y):
    return y.sum() / y.shape[0]


cdef double _var(np.ndarray[double, ndim=1] y):
    cdef double y_mean = _mean(y)
    cdef double diff_sum = 0.0

    for ele in y:
        diff_sum += (ele - y_mean) * (ele - y_mean)

    return diff_sum / y.shape[0]


# Returns negative mean-squared error
cdef double negative_mse(np.ndarray[double, ndim=1] y_true,
                         np.ndarray[double, ndim=1] y_pred):
    cdef np.ndarray[double, ndim=1] err
    err = y_true - y_pred

    cdef double squared_err_sum = 0.0
    cdef double ele
    for ele in err:
        squared_err_sum += ele * ele
    return squared_err_sum / y_true.shape[0]
    # return np.mean(np.square(y_true - y_pred))


# Returns negative least-absolute deviation
cdef double negative_lad(np.ndarray[double, ndim=1] y_true,
                         np.ndarray[double, ndim=1] y_pred):
    cdef np.ndarray[double, ndim=1] err
    err = y_true - y_pred

    cdef double absolute_err_sum = 0.0
    cdef double ele
    for ele in err:
        absolute_err_sum += abs(ele)

    return absolute_err_sum / y_true.shape[0]


# Returns the R-squared measure
cdef double rsquared(np.ndarray[double, ndim=1] y_true,
                     np.ndarray[double, ndim=1] y_pred):

    return 1.0 + negative_mse(y_true, y_pred) / _var(y_true)


# Returns the % of signs matched between two arrays
cdef double sign_match(np.ndarray[double, ndim=1] y_true,
                       np.ndarray[double, ndim=1] y_pred):
    cdef np.ndarray[double, ndim=1] sign_match
    sign_match = (y_true.sign() == y_pred.sign())
    return sign_match.sum() / y_true.shape[0]
