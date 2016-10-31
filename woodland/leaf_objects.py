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

import collections

from ..core import LeafDictionary


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
        self.__path = path
        self.__value = value
        self.__n_samples = n_samples

        self.__str_cache = None  # used to cache the string representation later

    def __str__(self):
        if self.__str_cache is None:
            node_strings = []
            keys = ["path", "value", "n_samples"]
            values = [self.path, self.value, self.n_samples]

            for key, val in zip(keys, values):
                node_strings.append("{}: {}".format(key, val))
            self.__str_cache = "({})".format(', '.join(node_strings))

        return self.__str_cache

    def __repr__(self):
        return self.__str__()

    @property
    def path(self):
        return self.__path

    @property
    def value(self):
        return self.__value

    @property
    def n_samples(self):
        return self.__n_samples


class LucidSKTree(LeafDictionary):
    """ Object representation of the unraveled leaf nodes of a decision tree
        It is essentially a wrapper around a dictionary.

        This object is intended to be created through unravel_tree only.

    ..note:
        The index of the leaf in the passed dictionary (tree_leaves) should be the index of
        the leaf in the pre-order traversal of the decision tree.
        The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)

        Shouldn't inherit from this class
    """

    def __init__(self, tree_leaves, print_limit=30):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
        """
        if not isinstance(tree_leaves, dict):
            raise ValueError("A dictionary object with keys mapped to SKTreeNodes should ",
                             "be passed into the constructor.")
        # Check all the values mapped are SKTreeNodes
        assert all(map(lambda x: isinstance(x, SKTreeNode), tree_leaves.values()))
        super(LucidSKTree, self).__init__(tree_leaves, print_limit)


class SKFoliage(LeafDictionary):
    """ SKFoliage is an object to represent the aggregated leaves returned by
        one of the aggregate functions ...
        (i.e. aggregate_activated_leaves, aggregate_trained_leaves)

    ..note: Shouldn't inherit from this class
    """

    def __init__(self, tree_leaves, print_limit=30):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
        """
        if not isinstance(tree_leaves, dict):
            raise ValueError("A dictionary object with keys mapped to lists of values should ",
                             "be passed into the constructor.")
        assert all(map(lambda x: isinstance(x, collections.Iterable), tree_leaves.values()))

        super(LucidSKTree, self).__init__(tree_leaves, print_limit)
