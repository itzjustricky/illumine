"""
    Description:
        Methods for analyzing tree nodes following the Scikit-Learn API

    TODO:
        * add more to compress the LucidEnsemble;
            there is a lot of inefficiency there
            - compress by identical trees instead of
                by unique nodes?
            - map feature_names to column index and
                convert DataFrame into a numpy matrix

    @author: Ricky
"""

import logging
from copy import deepcopy
# from collections import Iterable
# from collections import OrderedDict

import numpy as np
import pandas as pd

from ..tree.lucid_tree import LucidTree
from .compression import compress_leaves

__all__ = ['LucidEnsemble']

logger = logging.getLogger(__name__)


class LucidEnsemble(object):
    """ Object representation of an ensemble of unraveled decision trees
        It is essentially a wrapper around a list where the ...

        This class is NOT meant to be INHERITED from.
    """

    def __init__(self, lucid_trees, init_estimator, learning_rate):
        """ Construct the LucidTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)

        :param tree_ensemble (list): a list of the LucidTree objects
        :param init_estimator (function): function that is the initial estimator of the ensemble
        :param create_deepcopy (bool): indicates whether or not to make a deepcopy of
            the tree_ensemble argument passed into the __init__ function
        """
        # Check all the values mapped are LucidTrees
        assert all(map(lambda x: isinstance(x, LucidTree), lucid_trees))

        self._estimators = lucid_trees
        self._learning_rate = learning_rate
        self._n_estimators = len(lucid_trees)

        if hasattr(init_estimator, 'predict'):
            self._init_estimator = deepcopy(init_estimator)
        else:
            raise ValueError(
                "The init_estimator should be an object with a predict "
                "function with function signature predict(self, X) "
                "where X is the feature matrix.")

    @property
    def n_estimators(self):
        """ The # of tree-estimators in the ensemble """
        return self._n_estimators

    @property
    def learning_rate(self):
        return self._learning_rate

    def __reduce__(self):
        return (self.__class__, (
            self._seq,
            self._init_estimator,
            self._learning_rate)
        )

    def output_compressed_ensemble(self):
        leaf_table = compress_leaves(
            [lucid_tree._leaf_table for lucid_tree in self._estimators],
            self.learning_rate)
        return CompressedEnsemble(leaf_table, self._init_estimator)

    def apply(self, X):
        """ Apply estimators in Tree to a pandas DataFrame.
            The DataFrame columns should be the same as
            the feature_names attribute.
        """
        apply_matrix = np.zeros(
            (X.shape[0], self.n_estimators), dtype=np.int32)
        for ind, lucid_tree in enumerate(self._estimators):
            apply_matrix[:, ind] = lucid_tree.apply(X)

        return apply_matrix

    def predict(self, X):
        """ Create predictions from a pandas DataFrame.
            The DataFrame columns should be the same as
            the feature_names attribute.
        """
        y_pred = np.zeros(X.shape[0])
        for lucid_tree in self._estimators:
            y_pred += lucid_tree.predict(X)

        return y_pred * self.learning_rate + \
            self._init_estimator.predict(X).ravel()


class CompressedEnsemble(object):

    def __init__(self, leaf_table, init_estimator):
        self._leaf_table = leaf_table
        self._init_estimator = init_estimator

    def apply(self, X):
        """ Apply leaves in the tree to X, return leaf's index as an order
            in the Tree (not pre-order index, index is in [0, n_leaves])
        """
        if len(self) == 1:
            return np.zeros(X.shape[0], dtype=np.int32)
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(dtype=np.float64, order='F')

        return self._leaf_table.apply(X)

    def predict(self, X):
        """ Create predictions from a matrix of the feature variables """

        # this indicates the trained tree had no splits;
        # possible via building LucidTree from sklearn model
        if len(self) == 1:
            return np.zeros(X.shape[0])
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(dtype=np.float64, order='F')

        return self._leaf_table.predict(X) + \
            self._init_estimator.predict(X).ravel()

    def combine_with(self, other):
        pass

    def __len__(self):
        return len(self._leaf_table)

    def __reduce__(self):
        return (self.__class__, (
            tuple((self._leaf_table,))
        ))
