"""
    Description:
        Methods for analyzing tree nodes following the Scikit-Learn API

    TODO:
        * consider allowing n_jobs parameter to send out independent jobs
            to process the estimators separately
        * extend base_adjustment for unravel_tree to allow vector

    @author: Ricky
"""

import numpy as np

import operator
import itertools
from collections import OrderedDict

from .leaf_objects import SKTreeNode
from .leaf_objects import SKFoliage
from .leaf_objects import LucidSKTree
from .leaf_objects import LucidSKEnsemble

__all__ = ['unravel_tree', 'unravel_ensemble', 'get_tree_predictions',
           'aggregate_trained_leaves', 'aggregate_activated_leaves', 'get_top_leaves']


def unravel_tree(sk_tree, feature_names=None, display_relation=True,
                 base_adjustment=0, float_precision=3, sort_by_index=True,
                 tree_kw=None):
    """ Breakdown a tree's splits and returns the value of every leaf along
        with the path of splits that led to the leaf

    ..note:
        Scikit-learn represent their trees with nodes (represented by numbers) printed
        by preorder-traversal; number of -2 represents a leaf, the other numbers are by
        the index of the column for the feature

    :param feature_names (list): list of names (strings) of the features that were used
        to split the tree
    :param sk_tree: scikit-learn tree object
    :param display_relation (bool): if marked false then only display feature else display
        the relation as well; if marked true, the path
    :param base_adjustment (numeric): shift all the values with a base value
    :param float_precision (int): to determine what number the node values, thresholds are
        rounded to
    :param tree_kw (dict): key-word arguments to be passed into LucidSKTree's constructor

    :returns: LucidSKTree object indexed by their order in the
        pre-order traversal of the Decision Tree
    """
    if tree_kw is None:
        tree_kw = dict()
    elif not isinstance(tree_kw, dict):
        raise ValueError("tree_kw should be of type dict")
    tree_leaves = OrderedDict()

    values = sk_tree.tree_.value
    features = sk_tree.tree_.feature
    node_samples = sk_tree.tree_.n_node_samples
    thresholds = sk_tree.tree_.threshold

    n_splits = len(features)
    if n_splits == 0:
        raise ValueError("The passed tree is empty!")

    if feature_names is None:
        feature_names = np.arange(features.max() + 1)

    tracker_stack = []  # a stack to track if all the children of a node is visited
    node_index, node_path = 0, []  # ptr_stack keeps track of nodes
    for node_index in range(n_splits):
        if len(tracker_stack) != 0:
            tracker_stack[-1] += 1  # visiting the child of the latest node

        if features[node_index] != -2:  # visiting inner node
            tracker_stack.append(0)
            if display_relation:
                append_str = "{}<={}".format(feature_names[features[node_index]],
                                             float(round(thresholds[node_index], float_precision)))
            else:
                append_str = feature_names[features[node_index]]
            node_path.append(append_str)
        else:  # visiting leaf
            tree_leaves[node_index] = \
                SKTreeNode(node_path.copy(),
                           base_adjustment + float(round(values[node_index][0][0], float_precision)),
                           node_samples[node_index])

            if node_index in sk_tree.tree_.children_right:
                # pop out nodes that I am completely done with
                while(len(tracker_stack) > 0 and tracker_stack[-1] == 2):
                    node_path.pop()
                    tracker_stack.pop()
            if display_relation and len(node_path) != 0:
                node_path[-1] = node_path[-1].replace("<=", ">")

    return LucidSKTree(tree_leaves, **tree_kw)


def unravel_ensemble(sk_ensemble, tree_kw=None, ensemble_kw=None, **kwargs):
    """ Breakdown a tree's splits and returns the value of every leaf along
        with the path of splits that led to the leaf

    ..note:
        Scikit-learn represent their trees with nodes (represented by numbers) printed
        by preorder-traversal; number of -2 represents a leaf, the other numbers are by
        the index of the column for the feature

    :param sk_ensemble: scikit-learn ensemble model object
    :param feature_names (list): list of names (strings) of the features that were used
        to split the tree
    :param display_relation (bool): if marked false then only display feature else display
        the relation as well; if marked true, the path
    :param base_adjustment (numeric): shift all the values with a base value
    :param float_precision (int): to determine what number the node values, thresholds are
        rounded to
    :param tree_kw (dict): key-word arguments to be passed into LucidSKTree's constructor
    :param ensemble_kw (dict): key-word arguments to be passed into LucidSKEnsemble's constructor

    :returns: dictionary of SKTreeNode objects indexed by their order in the
        pre-order traversal of the Decision Tree
    """
    if tree_kw is None:
        tree_kw = dict()
    elif not isinstance(tree_kw, dict):
        raise ValueError("tree_kw should be of type dict")
    if ensemble_kw is None:
        ensemble_kw = dict()
    elif not isinstance(ensemble_kw, dict):
        raise ValueError("tree_kw should be of type dict")

    ensemble_of_leaves = []
    for estimator in sk_ensemble.estimators_:
        estimator = estimator[0]
        ensemble_of_leaves.append(unravel_tree(estimator, tree_kw=tree_kw, **kwargs))

    return LucidSKEnsemble(ensemble_of_leaves, **ensemble_kw)


def get_tree_predictions(sk_ensemble, X, adjust_with_base=False):
    """ Retrieve the tree predictions of each tree in the ensemble

    :param sk_ensemble: scikit-learn tree object
    :param X: array_like or sparse matrix, shape = [n_samples, n_features]
    :param adjust_with_init (bool): whether or not to adjust with the base/initial
        estimator; by default, in most sklearn ensemble objects, the first prediction
        is the mean of the target in the training data
    """
    if adjust_with_base:
        adjustment = sk_ensemble.init_.predict(X).ravel()
    else:
        adjustment = np.zeros(X.shape[0])

    leaf_values = np.zeros((X.shape[0], sk_ensemble.n_estimators))

    for ind, estimator in enumerate(sk_ensemble.estimators_):
        estimator = estimator[0]
        leaf_values[:, ind] = estimator.predict(X)

    return leaf_values + adjustment[:, np.newaxis]


def aggregate_trained_leaves(sk_ensemble, feature_names):
    """ Iterate through all the leaves from the trained trees in an ensemble
        and aggregate their values according to their paths as key values
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

    return SKFoliage(leaf_dict)


def aggregate_activated_leaves(sk_ensemble, X, feature_names):
    """ Iterate through the leaves activated from the data X and aggregate their
        values according to their paths as key values
    """
    leaf_dict = dict()

    # Get a matrix of all the leaves activated
    all_activated_leaves = sk_ensemble.apply(X)
    unraveled_ensemble = \
        unravel_ensemble(sk_ensemble, feature_names=feature_names, display_relation=True)

    # Iterate through the activated leaves for each data sample
    for active_leaves in all_activated_leaves:

        for estimator_ind, active_leaf_ind in enumerate(active_leaves):
            active_leaf = unraveled_ensemble[estimator_ind][active_leaf_ind]
            leaf_dict.setdefault(active_leaf.path.__str__(), []) \
                     .append(active_leaf.value)

    return SKFoliage(leaf_dict)


# I want to generalize this to get any measure per sample
def unique_leaves_per_sample(sk_ensemble, X, feature_names, scale_by_total=True):
    """ Iterate through the samples of data X and count the number
        of unique leaf paths activated

    :param sk_ensemble: scikit-learn ensemble model object
    :param X: the feature matrix (Nxp) where N and p is the # of samples and
        features, respectively
    :param feature_names (list): list of names (strings) of the features that were used
        to split the tree
    :param scale_by_total (bool): indicate whether or not to scale by the total number
        of unique leaves in the sk_ensemble
    """
    leaf_dict = dict()

    # Get a matrix of all the leaves activated
    all_activated_leaves = sk_ensemble.apply(X)
    unraveled_ensemble = \
        unravel_ensemble(sk_ensemble, feature_names=feature_names, display_relation=True)

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


def get_top_leaves(foliage_obj, n_top=50, rank='abs_sum'):
    """ Gather the top n leaves according to some rank function

    :param sk_ensemble: scikit-learn ensemble model object
    :param feature_names (list): list of names (strings) of the features that were used
        to split the tree
    :param foliage_obj: an instance of SKFoliage that is outputted from
        aggregate_trained_leaves or aggregate_activated_leaves methods
    :param n_top: the number of leaves to display
    :param rank: the ranking method for the leafpaths
    """
    if not isinstance(foliage_obj, SKFoliage):
        raise ValueError("The foliage_obj passed is of type {}".format(type(foliage_obj)),
                         "; it should be an instance of SKFoliage")
    valid_rank = {
        'abs_sum': lambda x: np.sum(np.abs(x)),
        'abs_mean': lambda x: np.mean(np.abs(x)),
        'mean': np.mean,
        'sum': np.sum,
        'std': np.std,
        'count': len}

    if isinstance(rank, str):
        if rank not in valid_rank.keys():
            raise ValueError("The passed rank ({}) argument is not a valid str {}"
                             .format(rank, list(valid_rank.keys())))
        rank = valid_rank[rank]
    elif not callable(rank):
        raise ValueError("The passed rank argument should be a callable function ",
                         "taking a vector as an argument or a valid str {}"
                         .format(list(valid_rank.keys())))

    aggregated_ranks = []
    # Gather the ranks
    for leaf_path, values in foliage_obj.items():
        aggregated_ranks.append(
            (leaf_path, rank(values)))

    top_leaf_paths = \
        map(operator.itemgetter(0),
            sorted(aggregated_ranks, key=operator.itemgetter(1), reverse=True))
    return SKFoliage(OrderedDict(((path, foliage_obj[path]) for path in itertools.islice(top_leaf_paths, n_top))))
