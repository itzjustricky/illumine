"""
    Description:



    @author: Ricky Chang
"""

import operator
import numpy as np

from ..woodland.leaf_objects import SKFoliage
from ..woodland.node_methods import rank_leaves


def plot_leaf_rank(foliage_obj, rank_method='abs_sum', cumulative_plot=False,
                   considered_leaves=None,
                   plt_xlabel='Index of Sorted Leaves', plt_ylabel='',
                   plt_title='Plot of Sorted Rank'):
    """ Plot the cumulative sum of some rank of an ensemble

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


def plot_step_improvement():
    """
    """
    def staged_error_function(staged_preds, y_true, axis=1):
        # axis: 0 for row, 1 for column
        return np.mean(np.abs(staged_preds - y_true), axis=axis)

    import matplotlib.pyplot as plt
    pass
