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


class LeafDictionary(object):
    """ This is to serve as a base class for classes that wrap iterative objects
        with leaf decision paths as keys (and anything for values).

        This was created because many of the classes created in leaf_objects had
        the same underlying structure.
    """

    def __init__(self, tree_leaves, print_limit, create_deepcopy, str_kw=None):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
        """
        if not isinstance(tree_leaves, collections.Iterable):
            raise ValueError("The passed type {} is not iterable".format(type(tree_leaves)))
        if create_deepcopy:
            self._seq = deepcopy(tree_leaves)
        else:
            self._seq = tree_leaves
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
            self._set_dictionary_attr(self._seq)

    def _get_dictionary_attr(self, cls_dict):
        """ Method to create dictionary attributes to be set
            for LeafDictionary. The attributes will point
            get values from cls_dict.
        """
        def keys(self):
            return cls_dict.keys()

        def values(self):
            return cls_dict.values()

        def items(self):
            return cls_dict.items()

        return_attrs = ['keys', 'values', 'items']
        return [val for key, val in locals().items() if key in return_attrs]

    def _set_dictionary_attr(self, cls_dict):
        """ Set dictionary attributes for LeafDictionary """
        for dict_func in self._get_dictionary_attr(cls_dict):
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

    def __getitem__(self, key):
        self.__str_cache = None  # reset string cache
        return self._seq[key]

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.__len

    def __str__(self):
        if self.__str_cache is None:
            self.__str_cache = \
                print_seq(iter_seq=self._seq, print_limit=self.__print_limit,
                          strip_at=',', return_string=True, **self.__str_kw)

        return self.__str_cache
