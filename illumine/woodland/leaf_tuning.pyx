"""
    Cython module for the retrieve_leaf_path
    function used to build lucid-tree objects

    TODO:
        allow for pruning over a random subsample
        of prune candidates

"""

import logging

cimport cython
from cpython cimport array
# from cython.parallel import prange

import numpy as np
cimport numpy as cnp
from numpy.math cimport INFINITY
from scipy.sparse import diags

from ..metrics cimport score_functions


# define a function type
ctypedef double (*score_f)(double[:] y_true, double[:] y_pred) nogil
ctypedef double (*deriv_f)(double[:] y_true, double[:] y_pred, int order) nogil


def finetune_ensemble(compressed_ensemble,
                      cnp.ndarray[double, ndim=2] X,
                      cnp.ndarray[double, ndim=1] y_true,
                      cnp.ndarray[double, ndim=1] y_pred,
                      int n_tunes, int chunk_size=10,
                      str metric_name='mse', double zero_bounds=0.01,
                      double l2_coef=0.0):
    """ Finetunes the leaves of a CompressedEnsemble object

    :param metric_name: the name of the metric to be used (e.g. mse, mad)
    :param zero_bounds: is a threshold used to determine whether to
        drop a leaf from the ensemble
    :param l2_coef: the l2 penalty coefficient for the leaf weights
    """
    # get the score and derivative function
    cdef list valid_functions = ['mse']
    cdef score_f score_function
    cdef deriv_f deriv_function

    if metric_name == 'mse':
        score_function = score_functions.negative_mse
        deriv_function = score_functions.mse_derivative
    else:
        raise ValueError(
            "Invalid metric_name was passed score_function, metric_name must be in {}"
            .format(valid_functions))

    cdef list ensemble_leaves = list(compressed_ensemble.leaves)
    cdef list leaf_values = list(compressed_ensemble.values())
    cdef int n_leaves = len(ensemble_leaves)
    # compressed_ensemble
    # compute_activation(self, X_df, considered_paths=None)
    cdef cnp.ndarray[double, ndim=2] pred_matrix
    cdef cnp.ndarray[double, ndim=1] pred_adj           # adjustment made to y_pred
    pred_adj = np.zeros(y_pred.shape[0])
    # cdef activation_matrix
    # cdef pred_matrix

    cdef double best_score, best_tune_ind, best_leaf_weight
    best_tune_ind = -1
    best_score = 0.0            # will be assigned below
    best_leaf_weight = 0.0      # will be assigned below

    cdef int i, leaf_iter
    for i in range(n_tunes):
        activation_matrix

        # go through chunks of prediction at a time
        for leaf_iter in range(0, n_tunes, chunk_size):
            chunk_end = min(leaf_iter + chunk_size, n_leaves)
            predarr_chunk = diags(leaf_values[leaf_iter:chunk_end])
            # this code piece is deprecated
            activation_matrix = compressed_ensemble.apply(X, ensemble_leaves[leaf_iter:chunk_end])
            pred_matrix = activation_matrix.dot(predarr_chunk).toarray()

            # best proposed improvement from this chunk
            new_score, tune_ind, new_tuned_weight = \
                find_tune_candidate(y_true, y_pred, pred_matrix,
                                    score_function, deriv_function, l2_coef)
            # adjust the index here
            if tune_ind != -1 and best_score < new_score:
                best_score = new_score
                best_tune_ind = tune_ind + leaf_iter * chunk_size    # must adjust index
                best_leaf_weight = new_tuned_weight

                # get the leaf prediction of the leaf, will be used to adjust y_pred
                pred_adj = pred_matrix[:, tune_ind]

        if best_tune_ind != -1:  # -1 indicates no improvement
            # if the new leaf value proposed is small then drop it
            chosen_leaf = ensemble_leaves[best_tune_ind]

            # will drop leaf from ensemble
            if abs(new_tuned_weight) <= zero_bounds:
                compressed_ensemble.drop(chosen_leaf)
                ensemble_leaves = list(compressed_ensemble.leaves)
                leaf_values = list(compressed_ensemble.values())
                n_leaves = len(ensemble_leaves)

                y_pred -= pred_adj
            # will adjust leaf to a new weight
            else:
                compressed_ensemble[chosen_leaf] = new_tuned_weight

                # the new prediction will be adjusted by the
                # difference of the old leaf weight and the new one
                for row_iter in range(pred_adj.shape[0]):
                    if pred_adj[row_iter] != 0.0:
                        pred_adj[row_iter] = new_tuned_weight - pred_adj[row_iter]

                y_pred -= pred_adj
        else:
            break


cdef tuple find_tune_candidate(double[:] y_true, double[:] y_pred,
                               double[:, :] pred_chunk,
                               score_f score_function, deriv_f deriv_function,
                               double l2_coef):
    """ Find a candidate leaf whose tuning leads to the highest new score

    :param pred_chunk: pxn matrix containing the predictions of
        some of the leaves in the ensemble;
        p is the # of leaves in the chunk
        n is the # of data samples
    """
    cdef int n_leaves = pred_chunk.shape[1]
    cdef int n_samples = pred_chunk.shape[0]

    # tmp_y_pred, tmp_y_true will only be non-zero
    # when the leaf in the loop is activated
    cdef double[:] tmp_y_pred   # store y_pred values without leaf value
    cdef double[:] tmp_y_true   # store y_true values for rows where leaf is activated
    tmp_y_pred = np.zeros(n_leaves, dtype=np.float64)
    tmp_y_true = np.zeros(n_leaves, dtype=np.float64)

    # store the best score as the score obtained using current prediction
    cdef double best_score = score_function(y_true, y_pred)

    cdef int best_leaf_ind = -1
    cdef double best_leaf_weight = 0.0
    cdef double new_score

    cdef int leaf_ind, i                    # integers for iterating in loop
    cdef double leaf_d1_sum, leaf_d2_sum    # var to determine new leaf weight
    cdef double new_leaf_weight             # var to store new leaf weight
    for leaf_ind in range(n_leaves):
        # leaf_d1_sum, leaf_d2_sum = 0.0, 0.0

        for i in range(n_samples):
            if pred_chunk[i][leaf_ind] == 0.0:
                tmp_y_pred[i] = 0.0
                tmp_y_true[i] = 0.0
            else:
                tmp_y_pred[i] = y_pred[i] - pred_chunk[i][leaf_ind]
                tmp_y_true[i] = y_true[i]

        # only sum over the indices where the leaf is activated
        leaf_d1_sum = deriv_function(tmp_y_true, tmp_y_pred, order=1)
        leaf_d2_sum = deriv_function(tmp_y_true, tmp_y_pred, order=2)
        # this new leaf-weight 2nd-order optimal
        # approximation see the xgboost paper
        new_leaf_weight = -0.5 * (leaf_d1_sum**2) / (leaf_d2_sum + l2_coef)
        for i in range(n_samples):
            if pred_chunk[i][leaf_ind] != 0.0:
                tmp_y_pred[i] += new_leaf_weight
        new_score = score_function(y_true, tmp_y_pred)

        # update leaf to tune if there is a new best score
        if best_score < new_score:
            best_score = new_score
            best_leaf_ind = leaf_ind
            best_leaf_weight = new_leaf_weight

    return (best_score, best_leaf_ind, best_leaf_weight)
