"""
    Description:



    @author: Ricky Chang
"""

from copy import deepcopy
from ..util.printing import print_seq


class LeafDictionary(object):
    """ This is to serve as a base class for classes that wrap dictionaries
        with leaf decision paths as keys (and anything for values).

        This was created because many of the classes created in leaf_objects had
        the same underlying structure.
    """

    def __init__(self, tree_leaves, print_limit=30):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
        """

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
            self.__str_cache = \
                print_seq(iter_seq=self.__dict, print_limit=self.__print_limit,
                          return_string=True)

        return self.__str_cache
