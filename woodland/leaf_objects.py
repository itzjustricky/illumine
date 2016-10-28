"""
    Description:
        Methods for analyzing tree nodes following the Scikit-Learn API

    unravel_tree : function
    access_data_in_node : function

    Not sure if I want the below functions
    conditional_count : function
    conditional_mean : function

    TODO:
        * consider using Cython for doing the heavy load in unravel_tree
            function since there can be a lot of overhead for
            trees with more depth
        * write a __reduce__ function for SKTreeNode
        * extend base_adjustment for unravel_tree to allow vector

    @author: Ricky
"""

from copy import deepcopy
from collections import OrderedDict


class SKTreeNode(object):
    """ Object representation a single node of a decision tree

    ..note: a big reason I decided to store path, value, n_samples in a
         dictionary is so I have a nice and easy loop to return a representation
    """

    def __init__(self, path, value, n_samples):
        """
        :param path (list): the decision path to the node
        :param value (numeric): the value associated with the node
        :param n_samples (int): the number of samples that reach the node
        """
        self._node_repr = OrderedDict()
        self._node_repr['path'] = path
        self._node_repr['value'] = value
        self._node_repr['n_samples'] = n_samples

        self.__str_cache = None  # used to cache the string representation later

    def __str__(self):
        if self.__str_cache is None:
            node_strings = []
            for key, val in self._node_repr.items():
                node_strings.append("{}: {}".format(key, val))
            self.__str_cache = "({})".format(', '.join(node_strings))
        return self.__str_cache

    def __repr__(self):
        return self.__str__()

    @property
    def path(self):
        return self._node_repr['path']

    @property
    def value(self):
        return self._node_repr['value']

    @property
    def n_samples(self):
        return self._node_repr['n_samples']


class LucidSKTree(object):
    """ Object representation of the unraveled leaf nodes of a decision tree
        It is essentially a wrapper around a dictionary.

        This object is intended to be created through unravel_tree only.

    ..note:
        The index of the leaf in the passed dictionary (tree_leaves) should be the index of
        the leaf in the pre-order traversal of the decision tree.

        The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
    """

    def __init__(self, tree_leaves, print_limit=30):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
        """
        if not isinstance(tree_leaves, dict):
            raise ValueError("A dictionary object with keys mapped to SKTreeNodes should ",
                             "be passed into the constructor.")

        self.__dict = deepcopy(tree_leaves)
        self.__str_cache = None  # used to cache the string representation later
        self.__print_limit = print_limit  # limit for how many SKTreeNode objects to print
        self.__len = len(tree_leaves)

    def set_print_limit(self, print_limit):
        if not isinstance(print_limit, int):
            raise ValueError("The print_limit passed should be an integer.")
        self.__print_limit = print_limit
        self.__str_cache = None  # reset string cache

    def keys(self):
        return self.__dict.keys()

    def values(self):
        return self.__dict.values()

    def items(self):
        return self.__dict.items()

    def __getitem__(self, index):
        self.__str_cache = None  # reset string cache
        return self.__dict[index]

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.__len

    def __str__(self):
        if self.__str_cache is None:
            dict_len = len(self.__dict)
            n_top_display, n_bottom_display = dict_len, dict_len

            # if limit violated, print half of print_limit at top and bottom
            if len(self.__dict) > self.__print_limit:
                n_top_display = self.__print_limit // 2
                n_bottom_display = self.__print_limit - n_top_display

            str_list = []
            for cnt, item in enumerate(self.__dict.items()):
                key, val = item
                if cnt < n_top_display or cnt >= dict_len - n_bottom_display:
                    str_list.append("[{}] {}".format(key, val))
                elif cnt == n_top_display:
                    str_list.append("...")
            self.__str_cache = "\n".join(str_list)

        return self.__str_cache
