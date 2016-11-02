"""
    Description:
        Methods for analyzing tree nodes following the Scikit-Learn API

    TODO:


    @author: Ricky
"""

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
        It is essentially a wrapper around a dictionary where the ...

        key: The index of the leaf in the passed dictionary (tree_leaves) should be the index of
             the leaf in the pre-order traversal of the decision tree.
             The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
        value: The value is an SKTreeNode object

    ..note:
        This object is intended to be created through unravel_tree only.
        This class is NOT meant to be INHERITED from.
    """

    def __init__(self, tree_leaves, print_limit=30, create_deepcopy=True):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
        """
        if not isinstance(tree_leaves, dict):
            raise ValueError("A dictionary object with keys mapped to SKTreeNodes should ",
                             "be passed into the constructor.")
        # Check all the values mapped are SKTreeNodes
        assert all(map(lambda x: isinstance(x, SKTreeNode), tree_leaves.values()))
        super(LucidSKTree, self).__init__(
            tree_leaves, print_limit, create_deepcopy)


class LucidSKEnsemble(LeafDictionary):
    """ Object representation of an ensemble of unraveled decision trees
        It is essentially a wrapper around a list where the ...

        index: The index of a tree model in its order of the additive process
            of an ensemble.
        value: The value is LucidSKTree object

    ..note:
        This object is intended to be created through unravel_ensemble only.
        This class is NOT meant to be INHERITED from.
    """

    def __init__(self, ensemble_trees, print_limit=5, create_deepcopy=True):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
        """
        if not isinstance(ensemble_trees, list):
            raise ValueError("A list object with index (by order of Boosts) mapped to Tree Estimators ",
                             "should be passed into the constructor.")
        # Check all the values mapped are LucidSKTrees
        assert all(map(lambda x: isinstance(x, LucidSKTree), ensemble_trees))
        str_kw = {"print_format": "Estimator {}\n============\n{}",
                  "print_with_index": True}

        super(LucidSKEnsemble, self).__init__(
            ensemble_trees, print_limit, create_deepcopy, str_kw)


class SKFoliage(LeafDictionary):
    """ Object representation of unique leaf nodes in an ensemble/tree model mapped
            to some data associated with the leaf nodes.
        It is essentially a wrapper around a dict where the ...

        key: Is the path to the unique leaf node, i.e.
            string repr. of ['PTRATIO>18.75', 'DIS>1.301', 'AGE>44.9', 'TAX<=368.0']
            where PTRATIO, DIS, AGE, & TAX are feature names
        value: Can be anything that describes some characteristics of the leaf node.
            For example, aggregate_trained_leaves finds all the instances of a certain leaf path
                of a trained ensemble and aggregates the leaf values. The aggregate_activated_leaves
                function finds all the "activated" leaf paths over a given dataset and aggregates
                the leaf values of those activated.

    ..note:
        This class is similar to LucidSKTree, but they are given different names to
            highlight the logical differences. LucidSKTree is a mapping to SKTreeNodes
            while SKFoliage is a mapping to any attribute of a leaf node.
        The use of this class is largely for duck typing and for correct use of woodland methods.

        This class is NOT meant to be INHERITED from.
    """

    def __init__(self, tree_leaves, print_limit=30, create_deepcopy=True):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
        """
        if not isinstance(tree_leaves, dict):
            raise ValueError("A dictionary object with keys mapped to lists of values should ",
                             "be passed into the constructor.")
        str_kw = {"print_format": "path: {}\n{}"}

        super(SKFoliage, self).__init__(
            tree_leaves, print_limit, create_deepcopy, str_kw)
