"""
    Description:
        Cython metric functions so they
        can be called in Cython modules.

        Created for optimization purposes.

    @author: Ricky
"""

cimport cython


cdef double negative_mse(double[:] y_true, double[:] y_pred) nogil
""" Returns negative mean-squared error """

cdef double negative_mad(double[:] y_true, double[:] y_pred) nogil
""" Returns negative mean-absolute deviation """

cdef double rsquared(double[:] y_true, double[:] y_pred) nogil
""" Returns the R-squared measure """

cdef double sign_match(double[:] y_true, double[:] y_pred) nogil
""" Returns the % of signs matched between two arrays """
