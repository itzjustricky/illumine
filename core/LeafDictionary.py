"""
    Description:
        LeafDictionary serves as a base class for classes that

    TODO:
        * see if custom __reduce__ function is needed


    @author: Ricky Chang
"""

import types
import collections

from copy import deepcopy

from ..util.printing import print_seq


def _get_dictionary_attr():
    """ Used to return the attributes of a dictionary that wraps
        a dictionary

    READ HERE
    ..note: Notice that it requires that the class variable be called self._seq.
        Not meant to be used outside of this module.
    """

    def keys(self):
        return self._seq.keys()

    def values(self):
        return self._seq.values()

    def items(self):
        return self._seq.items()

    return locals()


class LeafDictionary(object):
    """ This is to serve as a base class for classes that wrap iterative objects
        with leaf decision paths as keys (and anything for values).

        This was created because many of the classes created in leaf_objects had
        the same underlying structure.
    """

    def __init__(self, tree_leaves, print_limit=30, str_kw=None):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
        """
        if not isinstance(tree_leaves, collections.Iterable):
            raise ValueError("The passed type {} is not iterable".format(type(tree_leaves)))

        self._seq = deepcopy(tree_leaves)
        self.__str_cache = None  # used to cache the string representation later
        self.__len = len(tree_leaves)
        self.__print_limit = print_limit  # limit for how many SKTreeNode objects to print

        if str_kw is None:
            self.__str_kw = dict()
        elif isinstance(str_kw, dict):
            self.__str_kw = str_kw
        else:
            raise ValueError("str_kw should be an instance of a dictionary")

        # If the passed sequence is a dictionary set dictionary attributes
        if isinstance(tree_leaves, dict):
            for dict_func in _get_dictionary_attr().values():
                setattr(self, dict_func.__name__, types.MethodType(dict_func, self))

    def set_print_limit(self, print_limit):
        if not isinstance(print_limit, int):
            raise ValueError("The print_limit passed should be an integer.")
        self.__print_limit = print_limit
        self.__str_cache = None  # reset string cache

    def __contains__(self, item):
        return self._seq.__contains__(item)

    def __iter__(self):
        return self._seq.__iter__()

    def __getitem__(self, index):
        self.__str_cache = None  # reset string cache
        return self._seq[index]

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.__len

    def __str__(self):
        if self.__str_cache is None:
            self.__str_cache = \
                print_seq(iter_seq=self._seq, print_limit=self.__print_limit,
                          return_string=True, **self.__str_kw)

        return self.__str_cache
