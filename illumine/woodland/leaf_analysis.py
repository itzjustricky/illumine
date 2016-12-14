"""
    Description:



    @author: Ricky Chang
"""

import operator
from collections import Iterable
from collections import OrderedDict

import numpy as np

from .leaf_objects import LeafDataStore
from .leaf_objects import LucidSKEnsemble
from .factory_methods import make_LucidSKEnsemble

__all__ = ['gather_leaf_values', 'rank_leaves', 'rank_leaves_per_point',
           'get_tree_predictions', 'unique_leaves_per_sample']


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
            path_str = ' & '.join(sorted(active_leaf.path))
            leaf_dict.setdefault(path_str, []) \
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
            point_lds = \
                _gather_leaf_values(lucid_ensemble, active_leaves.reshape(1, -1),
                                    considered_leaves, create_deepcopy=False)
            lds_list.append(point_lds)
        return lds_list


valid_rank_methods = {
    'absolute-sum': lambda x: np.sum(np.abs(x)),
    'abssolute-mean': lambda x: np.mean(np.abs(x)),
    'count': len}


def rank_leaves(lds_obj, rank_method, float_precision=5, n_top=10):
    """ Gather the n_top leaves according to some rank_method function

    :param lds_obj: an instance of LeafDataStore that is outputted from
        aggregate_trained_leaves or aggregate_tested_leaves methods
    :param n_top (int): the number of leaves to display
    :param rank_method: the ranking method for the leafpaths
    """
    if not isinstance(lds_obj, LeafDataStore):
        raise ValueError("The lds_obj passed is of type {}".format(type(lds_obj)),
                         "; it should be an instance of LeafDataStore")

    if isinstance(rank_method, str):
        if rank_method not in valid_rank_methods.keys():
            raise ValueError(' '.join((
                "The passed rank_method ({}) argument is not a valid string argument {}."
                .format(rank_method, list(valid_rank_methods.keys())),
                "A callable object can also be passed."))
            )
        rank_method = valid_rank_methods[rank_method]

    elif not callable(rank_method):
        raise ValueError(' '.join((
            "The passed rank_method argument should be a callable function",
            "taking a vector as an argument or a valid str {}"
            .format(list(valid_rank_methods.keys()))
        )))

    aggregated_ranks = []
    # Gather the ranks
    for leaf_path, values in lds_obj.items():
        aggregated_ranks.append(
            (leaf_path, round(rank_method(values), float_precision))
        )
    aggregated_rank = sorted(aggregated_ranks, key=operator.itemgetter(1), reverse=True)

    return LeafDataStore(OrderedDict(
        ((path, rank) for path, rank in aggregated_rank[:n_top])))


def rank_leaves_per_point(lds_list, rank_method, float_precision=5, n_top=10):
    """ Gather the n_top leaves according to some rank_method function
        per data-point

    :param lds_obj: an instance of LeafDataStore that is outputted from
        aggregate_trained_leaves or aggregate_tested_leaves methods
    :param n_top: the number of leaves to display
    :param rank_method: the ranking method for the leafpaths
    :param considered_leaves: a list of the leaves to be considered
        defaults to None; if None then all leaves will be considered
    """
    if isinstance(lds_list, dict):
        raise ValueError("The passed argument lds_list should not be a dict, "
                         "it should be a list of LeafDataStore objects outputted "
                         "by gather_leaf_values with gather_method=per-point.")

    if not isinstance(lds_list, Iterable):
        raise ValueError("The passed argument for lds_list "
                         "must be an iterable object")
    if not all(map(lambda x: isinstance(x, LeafDataStore), lds_list)):
        raise ValueError("The passed argument for lds_list must "
                         "be a list of LeafDataStore objects.")

    rank_list = []
    for lds_obj in lds_list:
        rank_list.append(
            rank_leaves(
                lds_obj=lds_obj,
                rank_method=rank_method,
                float_precision=float_precision,
                n_top=n_top)
        )
    return rank_list


def get_tree_predictions(sk_ensemble, X, adjust_with_base=False):
    """ Retrieve the tree predictions of each tree in the ensemble

    :param sk_ensemble: scikit-learn tree object
    :param X: array_like or sparse matrix, shape = [n_samples, n_features]
    :param adjust_with_init (bool): whether or not to adjust with the base/initial
        estimator; by default, in most sklearn ensemble objects, the first prediction
        is the mean of the target in the training data
    """
    if adjust_with_base: adjustment = sk_ensemble.init_.predict(X).ravel()
    else: adjustment = np.zeros(X.shape[0])

    leaf_values = np.zeros((X.shape[0], sk_ensemble.n_estimators))

    for ind, estimator in enumerate(sk_ensemble.estimators_):
        estimator = estimator[0]
        leaf_values[:, ind] = estimator.predict(X)

    return leaf_values + adjustment[:, np.newaxis]


def unique_leaves_per_sample(sk_ensemble, X, feature_names, scale_by_total=True):
    """ Iterate through the samples of data X and count the number
        of unique leaf paths activated

    :param sk_ensemble: scikit-learn ensemble model object
    :param X: the feature matrix (Nxp) where N and p is the # of samples and
        features, respectively
    :param feature_names (list): list of names (strings) of the features that
        were used to split the tree
    :param scale_by_total (bool): indicate whether or not to scale by the
        total number of unique leaves in the sk_ensemble
    """

    # Get a matrix of all the leaves activated
    all_activated_leaves = sk_ensemble.apply(X)
    unraveled_ensemble = \
        make_LucidSKEnsemble(sk_ensemble, feature_names=feature_names, display_relation=True)

    # Nx1 matrix (where N is the # of samples) with counts of unique leaves per sample
    X_leaf_counts = []
    # Iterate through the activated leaves for each data sample
    for active_leaves in all_activated_leaves:

        tmp_leaf_set = set()
        for estimator_ind, active_leaf_ind in enumerate(active_leaves):
            active_leaf = unraveled_ensemble[estimator_ind][active_leaf_ind]
            tmp_leaf_set.add(active_leaf.path.__str__())
        X_leaf_counts.append(len(tmp_leaf_set))

    return np.array(X_leaf_counts)