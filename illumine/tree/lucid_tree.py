"""


"""

import numpy as np
from pandas import DataFrame

from collections import Iterable

from .leaf_dictionary import LeafDictionary
from .predict_methods import create_apply
from .predict_methods import create_prediction

from functools import total_ordering

__all__ = ['LeafPath', 'TreeLeaf', 'LucidTree']


@total_ordering
class LeafPath(object):
    """ Object representation of the path to a leaf node """

    def __init__(self, tree_splits):
        """ The initializer for LeafPath

        :param tree_splits (list): a list of TreeSplit objects which
            is defined in the Cython module leaf_retrieval

            A TreeSplit represent a single split in feature data X.
        """
        self._tree_splits = tree_splits
        self._key = None

    @property
    def tree_splits(self):
        return self._tree_splits

    def __iter__(self):
        return self.tree_splits.__iter__()

    def __str__(self):
        return self.tree_splits.__str__()

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.tree_splits)

    @property
    def key(self):
        """ The key attribute is used for sorting and
            defining the hash of the LeafPath object
        """
        if self._key is None:
            self._key = ''
            # let the key be composed of the feature_names then
            # the numbers this allows more informative sorting
            for split in self.tree_splits:
                self._key += split.feature_name
            for split in self.tree_splits:
                self._key += split.relation
                self._key += str(round(
                    split.threshold, split.print_precision))

        return self._key

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        return self.key < other.key


class TreeLeaf(object):
    """ Object representation a leaf (terminal node) of a decision tree.
        Stores the path to the leaf and the value associated with the leaf.
    """

    def __init__(self, path, value, n_samples):
        """
        :param path (list): the decision path to the node
        :param value (numeric): the value associated with the node
        :param n_samples (int): the number of samples that reach the node
        """
        self._path = LeafPath(path)
        self._value = value
        self._n_samples = n_samples

        # used to cache the string representation later
        self._str_cache = None

    def __str__(self):
        if self._str_cache is None:
            node_strings = []
            keys = ["path", "value", "n_samples"]
            values = [self.path, self.value, self.n_samples]

            for key, val in zip(keys, values):
                node_strings.append("{}: {}".format(key, val))
            self._str_cache = "({})".format(', '.join(node_strings))

        return self._str_cache

    def __repr__(self):
        return self.__str__()

    @property
    def path(self):
        return self._path

    @property
    def value(self):
        return self._value

    @property
    def n_samples(self):
        return self._n_samples


class LucidTree(LeafDictionary):
    """ Object representation of the unraveled leaf nodes of a decision tree
        It is essentially a wrapper around a dictionary where the ...

        key: The index of the leaf in the passed dictionary (tree_leaves) should be the index of
             the leaf in the pre-order traversal of the decision tree.
             The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
        value: The value is an TreeLeaf object

    ..note:
        This object is intended to be created through make_LucidTree only.
        This class is NOT meant to be INHERITED from.
    """

    def __init__(self, tree_structure, tree_leaves, feature_names, print_limit=30):
        """ Construct the LucidTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)

        :param tree_leaves (dict): a dictionary of representations of the leaves from the
            Scikit-learn tree models, the keys are the index of the leaves in the pre-order
            traversal of the decision tree
        :param feature_names (list): list of names (strings) of the features that
            were used to split the tree
        :param print_limit (int): configuration for how to print the LucidEnsemble
            out to the console
        """
        if not isinstance(tree_leaves, dict):
            raise ValueError("A dictionary object with keys mapped to TreeLeafs ",
                             "should be passed into the constructor.")
        # Check all the values mapped are TreeLeafs
        assert all(map(lambda x: isinstance(x, TreeLeaf), tree_leaves.values()))

        if not isinstance(feature_names, Iterable):
            raise ValueError(
                "feature_names should be an iterable object containing the "
                "feature names that the tree was trained on")
        self._feature_names = feature_names
        self._tree_structure = tree_structure

        super(LucidTree, self).__init__(
            tree_leaves,
            print_limit=print_limit)

    @property
    def feature_names(self):
        return self._feature_names

    def apply(self, X):
        """ Apply trees in Tree to a pandas DataFrame.
            The DataFrame columns should be the same as
            the feature_names attribute.
        """
        activated_indices = np.zeros(X.shape[0], dtype=int)
        # this indicates the trained tree had no splits
        # this is possible in Scikit-learn
        if len(self) == 1:
            return activated_indices

        if isinstance(X, DataFrame):
            # X = X.values
            X = X.values.astype(dtype=np.float64, order='F')
        leaf_paths, leaf_indices = \
            zip(*[(leaf.path, leaf_ind) for leaf_ind, leaf in self.items()])

        return create_apply(X, leaf_paths, leaf_indices)

    def predict(self, X):
        """ Create predictions from a pandas DataFrame.
            The DataFrame columns should be the same as
            the feature_names attribute.
        """
        y_pred = np.zeros(X.shape[0])
        # this indicates the trained tree had no splits
        # this is possible in Scikit-learn
        if len(self) == 1:
            return y_pred

        if isinstance(X, DataFrame):
            X = X.values.astype(dtype=np.float64, order='F')
        leaf_paths, leaf_values = \
            zip(*[(leaf.path, leaf.value) for leaf in self.values()])
        return create_prediction(X, leaf_paths, leaf_values)
        # activated_inds = self._tree_structure.decision_path(X)
        # for leaf_ind in self.keys():
        #     y_pred[np.where(activated_inds == leaf_ind)[0]] = self[leaf_ind].value
        # return y_pred

    def __reduce__(self):
        return (self.__class__, (
            self._tree_structure,
            self._seq,
            self._feature_names,
            self._print_limit)
        )
