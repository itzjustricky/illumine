
"""
    Cython metric functions so they
    can be called in Cython modules.

    Created for optimization purposes.

    @author: Ricky
"""

cimport cython

from libc.math cimport fabs, copysign


cdef double _calc_mean(double[:] vect) nogil:
    cdef int vect_size = vect.shape[0]
    cdef double vect_sum = 0.0

    for i in range(vect_size):
        vect_sum += vect[i]
    return vect_sum / vect_size


cdef double _calc_var(double[:] vect) nogil:
    cdef int vect_size = vect.shape[0]
    cdef double vect_mean = _calc_mean(vect)

    cdef double res = 0.0
    for i in range(vect_size):
        res += (vect[i] - vect_mean) ** 2
    return res / vect_size


cdef double negative_mse(double[:] y_true, double[:] y_pred) nogil:
    """ Returns negative mean-squared error """
    cdef int vect_size = y_true.shape[0]

    cdef double err = 0.0
    cdef double squared_sum = 0.0
    for i in range(vect_size):
        err = y_true[i] - y_pred[i]
        squared_sum -= (err * err)
    return squared_sum / vect_size


cdef double negative_mad(double[:] y_true, double[:] y_pred) nogil:
    """ Returns negative mean-absolute deviation """
    cdef int vect_size = y_true.shape[0]

    cdef double absolute_err_sum = 0.0
    for i in range(vect_size):
        absolute_err_sum -= fabs(y_true[i] - y_pred[i])

    return absolute_err_sum / vect_size


cdef double rsquared(double[:] y_true, double[:] y_pred) nogil:
    """ Returns the R-squared measure """

    return 1.0 + negative_mse(y_true, y_pred) / _calc_var(y_true)


cdef double sign_match(double[:] y_true, double[:] y_pred) nogil:
    """ Returns the % of signs matched between two arrays """
    cdef int vect_size = y_true.shape[0]

    cdef double sign_match_count = 0.0
    for i in range(vect_size):
        if copysign(1.0, y_true[i]) == copysign(1.0, y_pred[i]):
            sign_match_count += 1.0

    return sign_match_count / vect_size
