"""
    Cython metric functions so they
    can be called in Cython modules.

    Created for optimization purposes.

    @author: Ricky
"""

cimport cython


cdef double negative_mse(double[:] y_true, double[:] y_pred) nogil
""" Returns negative mean-squared error """

cdef double mse_derivative(double[:] y_true, double[:] y_pred, int order) nogil
""" Returns the derivative of mean-squared error function. Only supports
    first and second derivative.
"""

cdef double negative_mad(double[:] y_true, double[:] y_pred) nogil
""" Returns negative mean-absolute deviation """

cdef double mad_derivative(double[:] y_true, double[:] y_pred, int order) nogil
""" Returns the sum of the derivatives (wrt prediction) of mean-absolute
    deviation function. Only supports first derivative.
"""

cdef double rsquared(double[:] y_true, double[:] y_pred) nogil
""" Returns the R-squared measure """

cdef double sign_match(double[:] y_true, double[:] y_pred) nogil
""" Returns the % of signs matched between two arrays """
