"""
    Cython module for the retrieve_leaf_path
    function used to build lucid-tree objects

    TODO:
        allow for pruning over a random subsample
        of prune candidates

"""

import logging

cimport cython
from cython.parallel import prange
from cpython cimport array

import numpy as np
cimport numpy as cnp
from numpy.math cimport INFINITY

from ..core cimport metrics


# define a function type
ctypedef double (*metric_f)(double[:] y_true, double[:] y_pred) nogil


def find_prune_candidates(cnp.ndarray[double, ndim=1] y_true,
                          cnp.ndarray[double, ndim=1] y_pred,
                          cnp.ndarray[double, ndim=2] pred_matrix,
                          str metric_name,
                          int n_prunes):
    """ Used to find the columns to prune in the pred_matrix

    :param y_true: the values of the true target variables
    :param y_pred: the values of the outputted predicted
        target variables
    :param pred_matrix: a matrix of nxm of which each
        column is a component of the prediction
        (i.e. prediction for the jth datapoint is the sum of row j)
    :param metric_name: the name of the metric to be used to
        determine best fit
    :param n_prunes: the # of prunes to do. If the metric
        of the model does not improve, the pruning will ignore
        n_prunes and stop
    """
    cdef metric_f score_function

    cdef list valid_functions = [
        'mse', 'mad', 'rsquared', 'sign-match'
    ]

    if metric_name == 'mse':
        score_function = metrics.negative_mse
    elif metric_name == 'mad':
        score_function = metrics.negative_mad
    elif metric_name == 'rsquared':
        score_function = metrics.rsquared
    elif metric_name == 'sign-match':
        score_function = metrics.sign_match
    else:
        raise ValueError(
            "Invalid metric_name was passed, metric_name must be in {}"
            .format(valid_functions))

    return _find_prune_candidates(
        y_true,
        y_pred,
        pred_matrix,
        score_function,
        n_prunes)


cdef list _find_prune_candidates(double[:] y_true, double[:] y_pred,
                                 double[:, :] pred_matrix,
                                 metric_f score_function,
                                 int n_prunes):
    """ C-declared function to allow for the release
        of the gil for parallel computation
    """
    cdef int i, j, n_indices
    cdef int prune_ind                                  # index of candidate to be pruned
    cdef double global_best_score, local_best_score     # use to keep track of scores

    cdef int vect_size = y_true.shape[0]
    cdef double[:] y_pred_tmp                           # tmp storage of predictions
    cdef double[:] scores                               # store scores from pruning of candidates

    # keep a cython_view and python list object
    # of indices of prune candidates
    cdef int[:] cv_indices
    cdef list py_indices = list(range(pred_matrix.shape[1]))
    cdef list prune_candidates = []
    global_best_score = score_function(y_true, y_pred)

    y_pred_tmp = np.empty(vect_size, dtype=float)
    for prune_ind in range(n_prunes):

        cv_indices = array.array('i', py_indices)
        n_indices = cv_indices.shape[0]
        scores = np.zeros(n_indices)

        with nogil:
            for i in prange(n_indices):
                for j in range(vect_size):
                    y_pred_tmp[j] = y_pred[j] - pred_matrix[j, cv_indices[i]]

                scores[i] = score_function(
                    y_true=y_true,
                    y_pred=y_pred_tmp)

            # search for the index of the candidate
            # whose prune produces the best score
            prune_ind, local_best_score = 0, -INFINITY
            for i in range(n_indices):
                if scores[i] > local_best_score:
                    prune_ind = cv_indices[i]
                    local_best_score = scores[i]
        # end of nogil

        if global_best_score > local_best_score:
            return prune_candidates
        else:
            logging.getLogger(__name__).debug(
                "There are now {} prune candidate(s)"
                .format(len(prune_candidates)))

            for i in range(vect_size):
                y_pred[i] = y_pred[i] - pred_matrix[i, prune_ind]
            py_indices.remove(prune_ind)

            global_best_score = local_best_score
            prune_candidates.append(prune_ind)

    return prune_candidates
