"""
    Description:
        Some utilities for printing in a more compact way.
        Similar to how pandas printing works.

    TODO:
        * Extend to cut off horizontally as well

    @author: Ricky Chang
"""

import collections
import numpy as np


def print_dict(iter_dict, n_top_display, n_bottom_display, print_format, return_string):
    """ It is intended that this function is called from print_seq and not directly.

    :param iter_dict: an iterable object which you will print over
    :param n_top_display: the # of items to be displayed from the front of the list
    :param n_top_bottom: the # of items to be displayed from the bottom of the list
    :param print_format: describes the format in which the objects will be printed
    :param return_string (bool): if this is true then return a string instead of printing to console
    """
    seq_len = len(iter_dict)

    if print_format is None:
        print_format = "[{}] {}"
    elif not isinstance(print_format, str):
        raise ValueError("The print_format passed must be an instance of a string")

    str_list = []
    for cnt, item in enumerate(iter_dict.items()):
        key, val = item
        if cnt < n_top_display or cnt >= seq_len - n_bottom_display:
            str_list.append(print_format.format(key, val))
        elif cnt == n_top_display:
            str_list.append("...")
    return "\n".join(str_list)


def print_list(iter_list, n_top_display, n_bottom_display, print_format, return_string, print_with_index=False):
    """ It is intended that this function is called from print_seq and not directly.

    :param iter_list: an iterable object which you will print over
    :param n_top_display: the # of items to be displayed from the front of the list
    :param n_top_bottom: the # of items to be displayed from the bottom of the list
    :param print_format: describes the format in which the objects will be printed
    :param return_string (bool): if this is true then return a string instead of printing to console
    """
    seq_len = len(iter_list)

    if print_format is None:
        if print_with_index:
            print_format = "[{}] {}"
        else:
            print_format = "{}"
    elif not isinstance(print_format, str):
        raise ValueError("The print_format passed must be an instance of a string")

    str_list = []
    for ind, item in enumerate(iter_list):
        if ind < n_top_display or ind >= seq_len - n_bottom_display:
            if print_with_index: str_list.append(print_format.format(ind, item))
            else: str_list.append(print_format.format(item))
        elif ind == n_top_display:
            str_list.append("...")
    return "\n".join(str_list)


def print_seq(iter_seq, print_limit, print_format=None, return_string=False, **kwargs):
    """
    :param iter_seq: an iterable object which you will print over
    :param print_limit: the size limit threshold to indicate at which point items in the
        iterable object will be secluded from print
    :param print_format: describes the format in which the objects will be printed
    :param return_string (bool): if this is true then return a string instead of printing to console
    """
    if not isinstance(iter_seq, collections.Iterable):
        raise ValueError("iter_seq passed must be iterable!")

    dict_len = len(iter_seq)
    n_top_display, n_bottom_display = dict_len, dict_len

    # if limit violated, print half of print_limit at top and bottom
    if len(iter_seq) > print_limit:
        n_top_display = print_limit // 2
        n_bottom_display = print_limit - n_top_display

    if isinstance(iter_seq, dict):
        print_func = print_dict
    elif isinstance(iter_seq, list) or isinstance(iter_seq, np.ndarray):
        print_func = print_list
    else:
        raise ValueError("")

    if return_string:
        return print_func(iter_seq, n_top_display, n_bottom_display,
                          print_format, return_string, **kwargs)
    else:
        print_func(iter_seq, n_top_display, n_bottom_display,
                   print_format, return_string, **kwargs)
