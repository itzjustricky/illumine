"""
    Description:



    @author: Ricky Chang
"""

import operator
import numpy as np

from ..woodland.leaf_objects import SKFoliage
from ..woodland.node_methods import rank_leaves
from ..woodland.node_methods import get_tree_predictions


def plot_leaf_rank(foliage_obj, rank_method='abs_sum', cumulative_plot=False,
                   considered_leaves=None,
                   plt_xlabel='Index of Sorted Leaves', plt_ylabel='',
                   plt_title='Plot of Sorted Rank'):
    """ Plot the ascending ranks of the unique leaf nodes of an ensemble

    :param foliage_obj: an instance of SKFoliage that is outputted from
        aggregate_trained_leaves or aggregate_activated_leaves methods
    :param rank_method: the ranking method for the leafpaths
    :param cumulative_plot: indicates whether or not to plot the cumulative instead
        of the actual sorted rank
    :param considered_leaves: Default to None
        a list of the leaves to be considered; if None then all leaves will be considered
    """
    if not isinstance(foliage_obj, SKFoliage):
        raise ValueError("The foliage_obj passed is of type {}".format(type(foliage_obj)),
                         "; it should be an instance of SKFoliage")
    # TODO: I use this more than once, move somewhere else?
    valid_rank_methods = {
        'abs_sum': lambda x: np.sum(np.abs(x)),
        'abs_mean': lambda x: np.mean(np.abs(x)),
        'mean': np.mean,
        'sum': np.sum,
        'count': len}
    if isinstance(rank_method, str):
        if rank_method not in valid_rank_methods.keys():
            raise ValueError("The passed rank_method ({}) argument is not a valid str {}"
                             .format(rank_method, list(valid_rank_methods.keys())))
        rank_method = valid_rank_methods[rank_method]
    elif not callable(rank_method):
        raise ValueError("The passed rank_method argument should be a callable function ",
                         "taking a vector as an argument or a valid str {}"
                         .format(list(valid_rank_methods.keys())))

    if considered_leaves is None:
        n_top = len(foliage_obj)
    else:
        n_top = len(considered_leaves)

    # rank_leaves sorts leaf by rank internally (highest rank first)
    leaf_ranks = rank_leaves(foliage_obj, n_top=n_top, rank_method=rank_method,
                             return_type='rank', considered_leaves=considered_leaves)

    # change order to smallest rank first
    def retrieve_operator(dict_items):
        return list(reversed(
            list(map(operator.itemgetter(1), dict_items))))

    if cumulative_plot:
        plot_array = np.cumsum(retrieve_operator(leaf_ranks.items()))
    else:
        plot_array = np.array(retrieve_operator(leaf_ranks.items()))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(plot_array)
    ax.set_xlabel(plt_xlabel)
    ax.set_ylabel(plt_ylabel)
    ax.set_title(plt_title)

    return fig, ax


def plot_step_improvement(sk_ensemble, X, y, error_func=None,
                          plt_title=''):
    """ Plot the improvement per tree model added to the tree

    :param sk_ensemble: scikit-learn ensemble model object
    :param X: array_like or sparse matrix, shape = [n_samples, n_features]
        feature/dependent variables
    :param y: array_like or sparse matrix, shape = [n_samples, 1]
        target variable
    :param error_func: Defaults to None, if None it will take the error
        function from sk_ensemble.
        Calculates the error to be plotted. The function that takes
        arguments: pred, y

    """
    if not hasattr(sk_ensemble, 'staged_predict'):
        raise AttributeError("The sk_ensemble object passed does not have a staged_predict attribute")
    if error_func is None:
        error_func = sk_ensemble.loss_

    def stagewise_error(staged_preds, y_true, axis=1):
        errors = np.zeros(staged_preds.shape[0])
        for ind, col in enumerate(staged_preds):
            errors[ind] = error_func(col, y_true)
        return errors

    staged_preds = np.array(list(sk_ensemble.staged_predict(X)))
    staged_errors = stagewise_error(staged_preds, y)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(staged_errors)
    ax.set_xlabel('# of Estimators')
    ax.set_ylabel('Error')
    ax.set_title(plt_title)

    return fig, ax


def active_leaves_boxplot(sk_ensemble, X, plt_title=''):
    """ Use a boxplot to plot the distribution

    :param sk_ensemble: scikit-learn ensemble model object
    :param X: array_like or sparse matrix, shape = [n_samples, n_features]
        feature/dependent variables
    """

    n_samples = X.shape[0]
    tree_predictions = get_tree_predictions(sk_ensemble, X)

    # Function to create a list of lists into boxplot format
    def make_boxplot_ready(np_array):
        formatted_data = [np_array[i, :] for i in range(np_array.shape[0])]
        return formatted_data

    formatted_predictions = make_boxplot_ready(tree_predictions)
    import matplotlib.pyplot as plt

    # TODO: add many subplot functionality to visualize distribution across datarows
    fig, ax = plt.subplots(1)
    ax.boxplot(formatted_predictions, showfliers=False)
    ax.set_xticks(np.arange(1, n_samples + 1, 10))
    ax.set_xticklabels(np.arange(1, n_samples + 1, 10))
    ax.set_xlabel('Datarow #')
    ax.set_ylabel('Predictions')
    ax.set_title(plt_title)

    return fig, ax
