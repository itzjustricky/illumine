"""
    Cython module for the retrieve_leaf_path
    function used to build lucid-tree objects

"""

import logging

cimport cython
cimport numpy as cnp
from numpy.math cimport INFINITY

from ..core cimport metrics


# define a function type
ctypedef double (*metric_f)(double[:] y_true, double[:] y_pred)


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


cdef _find_prune_candidates(cnp.ndarray[double, ndim=1] y_true,
                            cnp.ndarray[double, ndim=1] y_pred,
                            cnp.ndarray[double, ndim=2] pred_matrix,
                            metric_f score_function,
                            int n_prunes):
    cdef list indices, prune_candidates
    indices = list(range(pred_matrix.shape[1]))
    prune_candidates = []

    cdef int prune_ind, worst_ind
    cdef double local_best_score, global_best_score, curr_score
    cdef cnp.ndarray[double, ndim=1] y_pred_tmp

    global_best_score = score_function(y_true, y_pred)
    for prune_ind in range(n_prunes):

        worst_ind, local_best_score = 0, -INFINITY
        # for ind in xrange(pred_matrix.shape[1]):
        for ind in indices:
            # if ind in prune_candidates:
            #     continue

            y_pred_tmp = y_pred - pred_matrix[:, ind]
            curr_score = score_function(
                y_true=y_true,
                y_pred=y_pred_tmp)

            if curr_score > local_best_score:
                logging.getLogger(__name__).debug(
                    "For prune {} the best score was updated to {}"
                    .format(prune_ind, curr_score))
                worst_ind = ind
                local_best_score = curr_score

        if global_best_score > local_best_score:
            return prune_candidates
        else:
            logging.getLogger(__name__).debug(
                "There are now {} prune candidate(s)"
                .format(len(prune_candidates)))

            y_pred = y_pred - pred_matrix[:, worst_ind]
            indices.remove(worst_ind)

            global_best_score = local_best_score
            prune_candidates.append(worst_ind)

    return prune_candidates
