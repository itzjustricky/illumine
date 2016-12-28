"""
    Description:



    @author: Ricky Chang
"""

import numpy as np
from pandas import DataFrame
from scipy.sparse import lil_matrix

from .leaf_objects import LeafDataStore
from .leaf_objects import LucidSKEnsemble

from .optimized_predict import map_features_to_int
from .optimized_predict import find_activated
from .factory_methods import make_LucidSKEnsemble

__all__ = ['gather_leaf_values', 'get_tree_predictions',
           'compute_activation']


# this method is not meant to be called outside this module
def _gather_leaf_values(lucid_ensemble, X_activated, considered_leaves=None, **lds_kw):
    """ Iterate through the leaves activated from the data X and gather
        their values according to their paths as key values

        The make_LucidSKEnsemble function is an expensive function to call, so
        this part of the function was separated away.

    :param lucid_ensemble: LucidSKEnsemble object which maps the tree index
        number to a LucidSKTree object; this tree is used to access data
        on the activated leaf nodes
    :param X_activated: (N x k) matrix
        where N, k are the # of samples, estimators respectively
        It represents the leaves activated per data sample
    :param considered_leaves: Default to None
        a list of the leaves to be considered; if None then all leaves
        will be considered
    """
    if not isinstance(lucid_ensemble, LucidSKEnsemble):
        raise ValueError("Argument lucid_ensemble should be an instance of LucidSKEnsemble")

    leaf_dict = dict()
    # Iterate through the activated leaves for each data sample
    for active_leaves in X_activated:

        for estimator_ind, active_leaf_ind in enumerate(active_leaves):
            active_leaf = lucid_ensemble[estimator_ind][active_leaf_ind]
            leaf_dict.setdefault(active_leaf, []) \
                     .append(active_leaf.value)

    if considered_leaves is None:
        return LeafDataStore(leaf_dict, **lds_kw)
    else:
        return LeafDataStore(dict((key, leaf_dict[key]) for key in considered_leaves),
                             **lds_kw)


def gather_leaf_values(sk_ensemble, X, feature_names, considered_leaves=None,
                       gather_method='aggregate', **lds_kw):
    """ This method is used to abstract away the _gather_leaf_values function so
        that a sk_ensemble and the original matrix data X is passed as arguments instead.

    :param sk_ensemble: scikit-learn ensemble model object
    :param feature_names (list): list of names (strings) of the features that
        were used to split the tree
    :param considered_leaves: Default to None
        a list of the leaves to be considered; if None then all leaves will be considered
    :param lds_kw: TODO
    """
    valid_gather_methods = ['aggregate',  # aggregate indicates to gather across all data samples
                            'per-point']  # per-sample indicates to gather values per point
    if gather_method not in valid_gather_methods:
        raise ValueError(' '.join((
            "The gather_method argument passed was not valid."
            "Valid arguments include [{}]"
            .format(' '.join(valid_gather_methods))
        )))

    # Get a matrix of all the leaves activated
    all_activated_leaves = sk_ensemble.apply(X)
    lucid_ensemble = make_LucidSKEnsemble(
        sk_ensemble, feature_names=feature_names,)

    if gather_method == 'aggregate':
        return _gather_leaf_values(
            lucid_ensemble, all_activated_leaves, considered_leaves, **lds_kw)
    else:  # gather_method == 'per-point'
        lds_list = []
        for active_leaves in all_activated_leaves:
            point_lds = _gather_leaf_values(
                lucid_ensemble, active_leaves.reshape(1, -1),
                considered_leaves, create_deepcopy=False)
            lds_list.append(point_lds)
        return lds_list


def get_tree_predictions(sk_ensemble, X, adjust_with_init=False):
    """ Retrieve the tree predictions of each tree in the ensemble

    :param sk_ensemble: scikit-learn tree object
    :param X: array_like or sparse matrix, shape = [n_samples, n_features]
    :param adjust_with_init (bool): whether or not to adjust with the
        initial estimator.

        By default, in most sklearn ensemble objects, the first
        prediction is the mean of the target in the training data
    """
    if adjust_with_init: adjustment = sk_ensemble.init_.predict(X).ravel()
    else: adjustment = np.zeros(X.shape[0])

    leaf_values = np.zeros((X.shape[0], sk_ensemble.n_estimators))

    for ind, estimator in enumerate(sk_ensemble.estimators_):
        estimator = estimator[0]
        leaf_values[:, ind] = estimator.predict(X)

    return leaf_values + adjustment[:, np.newaxis]


def compute_activation(lucid_ensemble, X_df, considered_leaves=None):
    """ Compute an activation matrix to be used as vectors for
        clustering leaves together.

        This function requires that lucid_ensemble is compressed, i.e.
        the compress() method was called.
    """
    if not isinstance(X_df, DataFrame):
        raise ValueError("The passed X_df argument should be of type DataFrame.")

    if not isinstance(lucid_ensemble, LucidSKEnsemble):
        raise ValueError("The passed lucid_ensemble argument should "
                         "be of type LucidSKEnsemble.")

    f_map = map_features_to_int(X_df.columns)
    X = X_df.values

    if considered_leaves is not None:
        if not all(map(lambda x: isinstance(x, str), considered_leaves)):
            raise ValueError(
                "All elements of considered_leaves should be of type string; "
                "the elements are string representations of the leaf-paths.")
        filtered_leaves = \
            dict(((key, lucid_ensemble.compressed_ensemble[key])
                  for key in considered_leaves))
    else:
        filtered_leaves = lucid_ensemble.compressed_ensemble

    leaf_paths = []
    activation_matrix = lil_matrix(
        (X_df.shape[0], lucid_ensemble.unique_leaves_count),
        dtype=bool)

    for ind, leaf_pair in enumerate(filtered_leaves.items()):
        path, value = leaf_pair

        leaf_paths.append(path)
        activated_indices = find_activated(X, f_map, path.split(' & '))
        activation_matrix[np.where(activated_indices)[0], ind] = 1

    return np.array(leaf_paths), activation_matrix.tocsr()
