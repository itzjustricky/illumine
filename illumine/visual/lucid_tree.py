"""
    Description:



    @author: Ricky Chang
"""

import operator

import numpy as np

from ..woodland.leaf_objects import LeafDataStore
from ..woodland.leaf_analysis import rank_leaves

__all__ = ['leaf_rank_plot', 'leaf_rank_barplot']


def leaf_rank_plot(foliage_obj, rank_method='abs_sum', considered_leaves=None,
                   plt_xlabel='Index of Sorted Leaves', plt_ylabel='',
                   plt_title='Plot of Ascending Rank'):
    """ Plot the ascending ranks of the unique leaf nodes of an ensemble

    :param foliage_obj: an instance of LeafDataStore that is outputted from
        aggregate_trained_leaves or aggregate_activated_leaves methods
    :param rank_method: the ranking method for the leafpaths
    :param considered_leaves: Default to None
        a list of the leaves to be considered; if None then all leaves will be considered
    """
    if not isinstance(foliage_obj, LeafDataStore):
        raise ValueError("The foliage_obj passed is of type {}".format(type(foliage_obj)),
                         "; it should be an instance of LeafDataStore")

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
    plot_array = np.array(retrieve_operator(leaf_ranks.items()))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(plot_array)
    ax.set_xlabel(plt_xlabel)
    ax.set_ylabel(plt_ylabel)
    ax.set_title(plt_title)

    return fig, ax


def leaf_rank_barplot(foliage_obj, n_top, rank_method='abs_sum', bar_color='#A2F789',
                      considered_leaves=None, path_output_file=None,
                      plt_xlabel='Index of Sorted Leaves', plt_ylabel='',
                      plt_title='Plot of Sorted Rank'):
    """ Plot the barplot of leaf rank

    :param path_output_file: TODO
    """

    if not isinstance(foliage_obj, LeafDataStore):
        raise ValueError("The foliage_obj passed is of type {}".format(type(foliage_obj)),
                         "; it should be an instance of LeafDataStore")

    # rank_leaves sorts leaf by rank internally (highest rank first)
    leaf_ranks = rank_leaves(foliage_obj, n_top=n_top, rank_method=rank_method,
                             return_type='rank', considered_leaves=considered_leaves)

    # change order to smallest rank first
    def retrieve_operator(dict_items):
        return list(reversed(
            list(map(operator.itemgetter(1), dict_items))))
    plot_array = np.array(retrieve_operator(leaf_ranks.items()))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(range(1, n_top + 1), plot_array, color=bar_color, align='center')
    ax.set_xlabel(plt_xlabel)
    ax.set_ylabel(plt_ylabel)
    ax.set_title(plt_title)

    if path_output_file is not None:
        with open(path_output_file, 'w') as out_file:
            out_file.write('Leaf paths\n')
            for key, val in leaf_ranks.items():
                out_file.write("{}, {:0.3f}\n".format(key, val))
    return fig, ax
