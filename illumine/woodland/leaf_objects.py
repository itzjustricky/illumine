"""
    Description:
        Methods for analyzing tree nodes following the Scikit-Learn API

    TODO:
        * add more to compress the LucidSKEnsemble;
            there is a lot of inefficiency there
            - compress by identical trees instead of
                by unique nodes?
            - map feature_names to column index and
                convert DataFrame into a numpy matrix

    @author: Ricky
"""

from copy import deepcopy
from collections import Iterable
from collections import OrderedDict
from functools import total_ordering

import numpy as np
from pandas import DataFrame

from ..core import LeafDictionary
from .optimized_predict import create_prediction

__all__ = ['LucidSKEnsemble', 'LucidSKTree']


@total_ordering
class LeafPath(object):
    """ Object representation of the path to a leaf node """

    def __init__(self, path):
        self._path = path

    def __str__(self):
        return self._path.__str__()

    def __repr__(self):
        return self.__str__()

    def __key(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __lt__(self, other):
        return hash(self) < hash(other)


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
        self._path = LeafPath(path)
        self._value = value
        self._n_samples = n_samples

        self._str_cache = None  # used to cache the string representation later

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


class LucidSKTree(LeafDictionary):
    """ Object representation of the unraveled leaf nodes of a decision tree
        It is essentially a wrapper around a dictionary where the ...

        key: The index of the leaf in the passed dictionary (tree_leaves) should be the index of
             the leaf in the pre-order traversal of the decision tree.
             The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
        value: The value is an SKTreeNode object

    ..note:
        This object is intended to be created through make_LucidSKTree only.
        This class is NOT meant to be INHERITED from.
    """

    def __init__(self, tree_leaves, feature_names, print_limit=30):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)

        :param tree_leaves (dict): a dictionary of representations of the leaves from the
            Scikit-learn tree models, the keys are the index of the leaves in the pre-order
            traversal of the decision tree
        :param feature_names (list): list of names (strings) of the features that
            were used to split the tree
        :param print_limit (int): configuration for how to print the LucidSKEnsemble
            out to the console
        """
        if not isinstance(tree_leaves, dict):
            raise ValueError("A dictionary object with keys mapped to SKTreeNodes should ",
                             "be passed into the constructor.")
        # Check all the values mapped are SKTreeNodes
        assert all(map(lambda x: isinstance(x, SKTreeNode), tree_leaves.values()))

        if not isinstance(feature_names, Iterable):
            raise ValueError(
                "feature_names should be an iterable object containing the "
                "feature names that the tree was trained on")
        self._feature_names = feature_names

        super(LucidSKTree, self).__init__(
            tree_leaves,
            print_limit=print_limit)

    @property
    def feature_names(self):
        return self._feature_names

    def predict(self, X_df):
        """ Create predictions from a pandas DataFrame.
            The DataFrame should have the same.
        """
        y_pred = np.zeros(X_df.shape[0])
        # this indicates the trained tree had no splits
        # this is possible in Scikit-learn
        if len(self) == 1:
            return y_pred

        if not isinstance(X_df, DataFrame):
            raise ValueError("Predictions must be done on a Pandas dataframe")
        if not all(X_df.columns == self.feature_names):
            raise ValueError("The passed pandas DataFrame columns should equal the "
                             "contain the feature_names attribute of the LucidSKTree")

        leaf_path, leaf_values = \
            zip(*[(leaf.path, leaf.value) for leaf in self.values()])

        return create_prediction(X_df, leaf_path, leaf_values)

    def __reduce__(self):
        return (self.__class__, (
            self._seq,
            self._feature_names,
            self._print_limit)
        )


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

    def __init__(self, tree_ensemble, feature_names, init_estimator,
                 learning_rate, print_limit=5):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)

        :param tree_ensemble (list): a list of the LucidSKTree objects
        :param feature_names (list): list of names (strings) of the features that
            were used to split the tree
        :param init_estimator (function): function that is the initial estimator of the ensemble
        :param print_limit (int): configuration for how to print the LucidSKEnsemble
            out to the console
        :param create_deepcopy (bool): indicates whether or not to make a deepcopy of
            the tree_ensemble argument passed into the __init__ function
        """
        if not isinstance(tree_ensemble, list):
            raise ValueError("A list object with index (by order of Boosts) mapped to Tree Estimators ",
                             "should be passed into the constructor.")
        # Check all the values mapped are LucidSKTrees
        assert all(map(lambda x: isinstance(x, LucidSKTree), tree_ensemble))
        self._feature_names = feature_names
        self._learning_rate = learning_rate
        self._unique_leaves_count = None
        self._total_leaves_count = None

        # this object will be created if compress method is called
        self._compressed_ensemble = None

        if not hasattr(init_estimator, 'predict'):
            raise ValueError(
                "The init_estimator should be an object with a predict "
                "function with function signature predict(self, X) "
                "where X is the feature matrix.")
        else:
            # this must be deepcopied to make LucidSKEnsemble pickeable
            self._init_estimator = deepcopy(init_estimator)

        str_kw = {"print_format": "Estimator {}\n============\n{}",
                  "print_with_index": True}

        super(LucidSKEnsemble, self).__init__(
            tree_ensemble,
            print_limit=print_limit,
            str_kw=str_kw)

    @property
    def feature_names(self):
        """ The name of the features that were used
            to train the scikit-learn model
        """
        return self._feature_names

    @property
    def total_leaves_count(self):
        """ The # of total leaves in the Ensemble, i.e. certain
            unique leaves may be counted more than once.
        """
        if self._total_leaves_count is None:
            cnt = 0
            for lucid_tree in self:
                cnt += len(lucid_tree)
            self._total_leaves_count = cnt
        return self._total_leaves_count

    @property
    def unique_leaves_count(self):
        """ The # of unique leaves in the Ensemble """
        if not self.is_compressed:
            raise AttributeError(
                "you must run the compress() method before "
                "getting unique_leaves_count.")
        else:
            return self._unique_leaves_count

    @property
    def is_compressed(self):
        """ Boolean to indicate whether or not the LucidSKEnsemble
            object is compressed or not.
        """
        return self._compressed_ensemble is not None

    @property
    def compressed_ensemble(self):
        """ The actual CompressedEnsemble object.
            The compress() method must be called before
            trying to get this object.
        """
        if self.is_compressed:
            return self._compressed_ensemble
        else:
            raise AttributeError(
                "you must run the compress() method before "
                "getting the compressed_ensemble.")

    def __reduce__(self):
        return (self.__class__, (
            self._seq,
            self.feature_names,
            self._init_estimator,
            self._learning_rate,
            self._print_limit)
        )

    def predict(self, X_df):
        if not isinstance(X_df, DataFrame):
            raise ValueError("Predictions must be done on a Pandas dataframe")
        if not all(X_df.columns == self.feature_names):
            raise ValueError("The passed pandas DataFrame columns should equal the "
                             "contain the feature_names attribute of the LucidSKTree")

        y_pred = np.zeros(X_df.shape[0])
        if self._compressed_ensemble is None:
            for lucid_tree in self:
                y_pred += lucid_tree.predict(X_df)

            return y_pred * self._learning_rate + self._init_estimator.predict(X_df).ravel()
        else:
            return self._compressed_ensemble.predict(X_df) \
                + self._init_estimator.predict(X_df).ravel()

    def compress(self, **kwargs):
        """ Create a CompressedEnsemble object which aggregates all
            the values of leaves with the same paths.

            This is useful if the # of unique leaves is smaller
            than the number of estimators.
        """
        unique_leaves = OrderedDict()
        for lucid_tree in self:

            # this indicates the trained tree had no splits
            # this is possible in Scikit-learn
            if len(lucid_tree) == 1:
                continue

            for leaf_node in lucid_tree.values():
                unique_leaves[leaf_node.path] = \
                    unique_leaves.get(leaf_node.path, 0) \
                    + self._learning_rate * leaf_node.value

        self._compressed_ensemble = CompressedEnsemble(
            unique_leaves, self.feature_names, **kwargs)
        self._unique_leaves_count = len(self._compressed_ensemble)


class LeafDataStore(LeafDictionary):
    """ Object representation of unique leaf nodes in an ensemble/tree model mapped
            to some data associated with the leaf nodes.
        It is essentially a wrapper around a dict where the ...

        key: the path to the unique leaf node. Example:
            The list representation might be
            ['PTRATIO>18.75', 'DIS>1.301', 'AGE>44.9', 'TAX<=368.0'],
            which will then be represented as a string in the following way
            'AGE>44.9 & DIS>1.301 & PTRATIO>18.75 & TAX<=368.0'.

            Where PTRATIO, DIS, AGE, & TAX are feature names

        value: Can be anything that describes some characteristics of the leaf node.
            For example, aggregate_trained_leaves finds all the instances of a certain leaf path
                of a trained ensemble and aggregates the leaf values. The aggregate_activated_leaves
                function finds all the "activated" leaf paths over a given dataset and aggregates
                the leaf values of those activated.

    ..note:
        This class is similar to LucidSKTree, but they are given different names to
            highlight the logical differences. LucidSKTree is a mapping to SKTreeNodes
            while LeafDataStore is a mapping to any attribute of a leaf node.
        The use of this class is largely for duck typing and for correct use of woodland methods.

        This class is NOT meant to be INHERITED from.
    """

    def __init__(self, tree_leaves, print_limit=30):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes
            (inner & leaf nodes)
        """
        if not isinstance(tree_leaves, dict):
            raise ValueError("A dictionary object with keys mapped to lists "
                             "of values should be passed into the constructor.")
        str_kw = {"print_format": "path: {}\n{}"}

        super(LeafDataStore, self).__init__(
            tree_leaves,
            print_limit=print_limit,
            str_kw=str_kw)

    def __reduce__(self):
        return (self.__class__, (
            self._seq,
            self._print_limit)
        )


class CompressedEnsemble(LeafDictionary):
    """ Compressed Ensemble Tree created from
        LucidSKEnsemble.compress() method
    """

    def __init__(self, tree_leaves, feature_names, print_limit=30):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
            This object should be constructed using LucidSKEnsemble.compress() method
        """
        if not isinstance(tree_leaves, dict):
            raise ValueError("A dictionary object with keys mapped to SKTreeNodes should ",
                             "be passed into the constructor.")

        if not isinstance(feature_names, Iterable):
            raise ValueError(
                "feature_names should be an iterable object containing the "
                "feature names that the tree was trained on")
        self._feature_names = feature_names

        super(CompressedEnsemble, self).__init__(
            tree_leaves, print_limit)

    @property
    def feature_names(self):
        """ The name of the features that were used
            to train the scikit-learn model
        """
        return self._feature_names

    def predict(self, X_df):
        """ Create predictions from a pandas DataFrame.
            The DataFrame should have the same.
        """
        if not isinstance(X_df, DataFrame):
            raise ValueError("Predictions must be done on a Pandas dataframe")
        if not all(X_df.columns == self.feature_names):
            raise ValueError("The passed dataframe should")

        leaf_path, leaf_values = \
            zip(*[(leaf_node.path, value) for leaf_node, value in self.items()])

        return create_prediction(X_df, leaf_path, leaf_values)
