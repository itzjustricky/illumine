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


def _enforce_horizontal_limit(string, horizontal_limit, strip_at):
    """ Method to slice a string if passes some horizontal limit

    :param string (str): the string to be
    :param horizontal_limit (int): the limit of horizontal
    :param strip_at (str): the string to start strip at
    """
    # do the method per line
    line_strings = string.split('\n')

    def apply_limit(string, horizontal_limit, strip_at):
        left_split_display = string[:(horizontal_limit // 2 + 1)]
        right_split_display = string[-(horizontal_limit // 2):]

        if strip_at is not None:
            strip_spot = left_split_display.rfind(strip_at) + 1
            left_split_display = left_split_display[:strip_spot]
            strip_spot = right_split_display.find(strip_at) + 1
            right_split_display = right_split_display[strip_spot:]

        if len(string) > horizontal_limit:
            string = "{} ... {}".format(
                left_split_display,
                right_split_display)
        return string

    for ind, line_string in enumerate(line_strings):
        line_strings[ind] = \
            apply_limit(line_string, horizontal_limit, strip_at)
    return '\n'.join(line_strings)


def print_dict(iter_dict, n_top_display, n_bottom_display, strip_at=None,
               print_format=None, horizontal_limit=100):
    """ Pretty print a dictionary with horizontal/vertical truncation.
        It is intended that this function is called from print_seq and not directly.

    :param iter_dict: an iterable object which you will print over
    :param n_top_display: the # of items to be displayed from the front of the list
    :param n_top_bottom: the # of items to be displayed from the bottom of the list
    :param print_format: describes the format in which the objects will be printed
    """
    seq_len = len(iter_dict)

    if print_format is None:
        print_format = "[{}] {}"
    elif not isinstance(print_format, str):
        raise ValueError("The print_format passed must be an instance of a string")

    string_list = []
    for cnt, item in enumerate(iter_dict.items()):
        key, val = item
        if cnt < n_top_display or cnt >= seq_len - n_bottom_display:
            string_list.append(_enforce_horizontal_limit(
                print_format.format(key, val), horizontal_limit, strip_at))
        elif cnt == n_top_display:
            string_list.append("...")
    return "\n".join(string_list)


def print_list(iter_list, n_top_display, n_bottom_display, strip_at=None,
               print_format=None, horizontal_limit=100, print_with_index=False):
    """ Pretty print a list with horizontal/vertical truncation.
        It is intended that this function is called from print_seq and not directly.

    :param iter_list: an iterable object which you will print over
    :param n_top_display: the # of items to be displayed from the front of the list
    :param n_top_bottom: the # of items to be displayed from the bottom of the list
    :param print_format: describes the format in which the objects will be printed
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

            if print_with_index:
                str_list.append(_enforce_horizontal_limit(
                    print_format.format(ind, item), horizontal_limit, strip_at))
            else:
                str_list.append(_enforce_horizontal_limit(
                    print_format.format(item), horizontal_limit, strip_at))
        elif ind == n_top_display:
            str_list.append("...")
    return "\n".join(str_list)


def print_seq(iter_seq, print_limit, print_format=None, return_string=False, **kwargs):
    """ Pretty print an iterable sequence with horizontal/vertical truncation

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
        raise ValueError(
            "print_seq function only supports printing "
            "types dict, lists, and np.ndarrays")

    if return_string:
        return print_func(iter_seq,
                          n_top_display=n_top_display,
                          n_bottom_display=n_bottom_display,
                          print_format=print_format,
                          **kwargs)
    else:
        print_func(iter_seq,
                   n_top_display=n_top_display,
                   n_bottom_display=n_bottom_display,
                   print_format=print_format,
                   **kwargs)
