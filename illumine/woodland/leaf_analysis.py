"""
    Description:



    @author: Ricky Chang
"""

import operator
import itertools
from collections import OrderedDict

import numpy as np

from .leaf_objects import SKFoliage
from .node_methods import unravel_tree
from .node_methods import unravel_ensemble

__all__ = ['aggregate_trained_leaves', 'aggregate_tested_leaves',
           'rank_leaves', 'rank_per_sample']


def _aggregate_tested_leaves(lucid_ensemble, X_activated, considered_leaves=None, **foliage_kw):
    """ Iterate through the leaves activated from the data X and aggregate their
        values according to their paths as key values

        The unravel_ensemble function is an expensive function to call, so
        this part of the function was separated away.

    :param lucid_ensemble: LucidSKEnsemble object which maps the tree index
        number to a LucidSKTree object; this tree is used to access data
        on the activated leaf nodes
    :param X_activated: (N x k) matrix
        where N, k are the # of samples, estimators respectively
        It represents the leaves activated per data sample
    :param considered_leaves: Default to None
        a list of the leaves to be considered; if None then all leaves will be considered
    """
    # considered_leaves_declared = !(considered_leaves is None)
    leaf_dict = dict()

    # Iterate through the activated leaves for each data sample
    for active_leaves in X_activated:

        for estimator_ind, active_leaf_ind in enumerate(active_leaves):
            active_leaf = lucid_ensemble[estimator_ind][active_leaf_ind]
            leaf_dict.setdefault(active_leaf.path.__str__(), []) \
                     .append(active_leaf.value)
    if considered_leaves is None:
        return SKFoliage(leaf_dict, **foliage_kw)
    else:
        return SKFoliage(dict(leaf_dict[key] for key in considered_leaves), **foliage_kw)


def aggregate_tested_leaves(sk_ensemble, X, feature_names, considered_leaves=None, **foliage_kw):
    """ This method is used to abstract _aggregate_tested_leaves method.
        Iterate through all the leaves from the trained trees in an ensemble
        and aggregate their values according to their paths as key values

    :param sk_ensemble: scikit-learn ensemble model object
    :param feature_names (list): list of names (strings) of the features that
        were used to split the tree
    :param considered_leaves: Default to None
        a list of the leaves to be considered; if None then all leaves will be considered
    :param foliage_kw: TODO
    """
    # Get a matrix of all the leaves activated
    all_activated_leaves = sk_ensemble.apply(X)
    lucid_ensemble = \
        unravel_ensemble(sk_ensemble, feature_names=feature_names, display_relation=True)

    return _aggregate_tested_leaves(
        lucid_ensemble, all_activated_leaves, considered_leaves, **foliage_kw)


def aggregate_trained_leaves(sk_ensemble, feature_names, considered_leaves=None, **foliage_kw):
    """ Iterate through all the leaves from the trained trees in an ensemble
        and aggregate their values according to their paths as key values

    :param sk_ensemble: scikit-learn ensemble model object
    :param feature_names (list): list of names (strings) of the features that
        were used to split the tree
    :param considered_leaves: Default to None
        a list of the leaves to be considered; if None then all leaves will be considered
    :param foliage_kw: TODO
    """
    # dictionary with path as the key mapped to a list of values
    leaf_dict = dict()

    for estimator in sk_ensemble.estimators_:
        estimator = estimator.ravel()[0]
        estimator_leaves = unravel_tree(estimator, feature_names,
                                        display_relation=True)

        for leaf in estimator_leaves.values():
            leaf_dict.setdefault(leaf.path.__str__(), []) \
                     .append(leaf.value)

    if considered_leaves is None:
        return SKFoliage(leaf_dict, **foliage_kw)
    else:
        return SKFoliage(dict(leaf_dict[key] for key in considered_leaves), **foliage_kw)


valid_rank_methods = {
    'abs_sum': lambda x: np.sum(np.abs(x)),
    'abs_mean': lambda x: np.mean(np.abs(x)),
    'count': len}


def rank_leaves(foliage_obj, n_top=10, rank_method='abs_sum', return_type='rank',
                considered_leaves=None):
    """ Gather the n_top leaves according to some rank_method function

    :param sk_ensemble: scikit-learn ensemble model object
    :param feature_names (list): list of names (strings) of the features that
        were used to split the tree
    :param foliage_obj: an instance of SKFoliage that is outputted from
        aggregate_trained_leaves or aggregate_tested_leaves methods
    :param n_top: the number of leaves to display
    :param rank_method: the ranking method for the leafpaths
    :param return_type (str): describes the value returned
    :param considered_leaves: Default to None
        a list of the leaves to be considered; if None then all leaves will be considered
    """
    if not isinstance(foliage_obj, SKFoliage):
        raise ValueError("The foliage_obj passed is of type {}".format(type(foliage_obj)),
                         "; it should be an instance of SKFoliage")

    valid_return_types = ['rank', 'values']
    if return_type not in valid_return_types:
        raise ValueError("The passed return_type ({}) is not valid".format(return_type),
                         " must be one of the following {}".format(valid_return_types))

    global valid_rank_methods
    if isinstance(rank_method, str):
        if rank_method not in valid_rank_methods.keys():
            raise ValueError("The passed rank_method ({}) argument is not a valid str {}"
                             .format(rank_method, list(valid_rank_methods.keys())))
        rank_method = valid_rank_methods[rank_method]
    elif not callable(rank_method):
        raise ValueError("The passed rank_method argument should be a callable function ",
                         "taking a vector as an argument or a valid str {}"
                         .format(list(valid_rank_methods.keys())))

    aggregated_ranks = []
    # Gather the ranks
    for leaf_path, values in foliage_obj.items():
        aggregated_ranks.append(
            (leaf_path, rank_method(values)))
    aggregated_rank = sorted(aggregated_ranks, key=operator.itemgetter(1), reverse=True)

    if return_type == 'rank':
        return SKFoliage(OrderedDict(
            ((path, rank) for path, rank in aggregated_rank[:n_top])))
    else:  # return_type == 'values'
        top_leaf_paths = map(operator.itemgetter(0), aggregated_rank)
        return SKFoliage(OrderedDict(
            ((path, foliage_obj[path]) for path in itertools.islice(top_leaf_paths, n_top))))


def rank_per_sample(sk_ensemble, X, feature_names, considered_leaves=None, **kwargs):
    """ Gather and rank the leaves activated per sample in the X dataset

    :param sk_ensemble: scikit-learn ensemble model object
    :param feature_names (list): list of names (strings) of the features that
        were used to split the tree
    :param X: array_like or sparse matrix, shape = [n_samples, n_features]
    :param n_top: the number of leaves to display
    :param rank_method: the ranking method for the leafpaths
    :param considered_leaves: Default to None
        a list of the leaves to be considered; if None then all leaves will be considered
    :returns: a list of SKFoliage objects that contain the n_top
        leaf nodes per sample according to some rank_method
    """
    # Get a matrix of all the leaves activated
    all_activated_leaves = sk_ensemble.apply(X)
    lucid_ensemble = \
        unravel_ensemble(sk_ensemble, feature_names=feature_names, display_relation=True)

    top_leaf_samples = []
    for active_leaves in all_activated_leaves:
        sample_foliage = \
            _aggregate_tested_leaves(lucid_ensemble, active_leaves.reshape(1, -1),
                                     considered_leaves=considered_leaves, create_deepcopy=False)
        top_leaf_samples.append(rank_leaves(foliage_obj=sample_foliage,
                                            return_type='rank', **kwargs))
    return top_leaf_samples
