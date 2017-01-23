"""
    Cython module for the retrieve_leaf_path
    function used to build lucid-tree objects

"""

cimport cython
import numpy as np
cimport numpy as np

from numpy.math cimport INFINITY

from ..core cimport metrics


# define a function type
ctypedef double (*metric_f)(np.ndarray[double, ndim=1] y_true,
                            np.ndarray[double, ndim=1] y_pred)


cdef sum_rows(np.ndarray[double, ndim=2] pred_matrix):
    cdef int n_rows, n_cols
    n_rows = pred_matrix.shape[0]
    n_cols = pred_matrix.shape[1]

    cdef np.ndarray[double, ndim=1] sum_vector = \
        np.zeros(n_rows, dtype=np.float64)

    for row in xrange(n_rows):
        for col in xrange(n_cols):
            sum_vector[row] += pred_matrix[row][col]
    return sum_vector


def find_prune_candidates(np.ndarray[double, ndim=1] y_true,
                          np.ndarray[double, ndim=1] y_pred,
                          np.ndarray[double, ndim=2] pred_matrix,
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
        'mse', 'lad', 'rsquared', 'sign-match'
    ]

    if metric_name == 'mse':
        score_function = metrics.negative_mse
    elif metric_name == 'lad':
        score_function = metrics.negative_lad
    elif metric_name == 'rsquared':
        score_function = metrics.rsquared
    elif metric_name == 'sign-match':
        score_function = metrics.sign_match
    else:
        raise ValueError(
            "An invalid metric_name was specified must be one of the following {}"
            .format(valid_functions))

    return _find_prune_candidates(
        y_true,
        y_pred,
        pred_matrix,
        score_function,
        n_prunes)


cdef _find_prune_candidates(np.ndarray[double, ndim=1] y_true,
                            np.ndarray[double, ndim=1] y_pred,
                            np.ndarray[double, ndim=2] pred_matrix,
                            metric_f score_function,
                            int n_prunes):
    cdef list prune_candidates
    prune_candidates = []

    cdef int prune_ind, worst_ind
    cdef double local_best_score, global_best_score, curr_score
    cdef np.ndarray[double, ndim=1] y_pred_tmp

    global_best_score = score_function(y_true, y_pred)

    for prune_ind in range(n_prunes):

        worst_ind, local_best_score = 0, -INFINITY
        for ind in xrange(pred_matrix.shape[1]):
            if ind in prune_candidates:
                continue

            # print(pred_matrix[:, prune_candidates + [ind]])
            y_pred_tmp = y_pred - \
                sum_rows(pred_matrix[:, prune_candidates + [ind]])

            curr_score = score_function(
                y_true=y_true,
                y_pred=y_pred_tmp)

            if curr_score > local_best_score:
                worst_ind = ind
                local_best_score = curr_score

        if global_best_score > local_best_score:
            return prune_candidates
        else:
            global_best_score = local_best_score
            prune_candidates.append(worst_ind)
    return prune_candidates
