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
from collections import Iterable
from collections import OrderedDict

import numpy as np
from scipy.sparse import lil_matrix
# from scipy.sparse import diags
from pandas import DataFrame

from ..tree.leaf_dictionary import LeafDictionary
from ..tree.lucid_tree import LucidTree
# from ..tree.lucid_tree import LeafPath
from ..tree.predict_methods import create_prediction
from ..tree.predict_methods import find_activated

from .find_prune_candidate import find_prune_candidates
from .leaf_tuning import finetune_ensemble
from ..utils.array_check import flatten_1darray

__all__ = ['LucidEnsemble', 'CompressedEnsemble']

logger = logging.getLogger(__name__)


def _prep_for_prediction(leaf_obj):
    """ Get the variables needed from a leaf-object to be
        passed into methods from the predict_methods module

    :param leaf_obj: TODO
    :returns: TODO
    """
    pass


class LucidEnsemble(LeafDictionary):
    """ Object representation of an ensemble of unraveled decision trees
        It is essentially a wrapper around a list where the ...

        index: The index of a tree model in its order of the additive process
            of an ensemble.
        value: The value is LucidTree object

    ..note:
        This object is intended to be created through unravel_ensemble only.
        This class is NOT meant to be INHERITED from.
    """

    def __init__(self, tree_ensemble, feature_names, init_estimator,
                 learning_rate, print_limit=30):
        """ Construct the LucidTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)

        :param tree_ensemble (list): a list of the LucidTree objects
        :param feature_names (list): list of names (strings) of the features that
            were used to split the tree
        :param init_estimator (function): function that is the initial estimator of the ensemble
        :param print_limit (int): configuration for how to print the LucidEnsemble
            out to the console
        :param create_deepcopy (bool): indicates whether or not to make a deepcopy of
            the tree_ensemble argument passed into the __init__ function
        """
        if not isinstance(tree_ensemble, list):
            raise ValueError("A list object with index (by order of Boosts) mapped to Tree Estimators ",
                             "should be passed into the constructor.")
        # Check all the values mapped are LucidTrees
        assert all(map(lambda x: isinstance(x, LucidTree), tree_ensemble))
        self._feature_names = feature_names
        self._learning_rate = learning_rate
        self._unique_leaves_count = None
        self._leaves_count = None

        if not hasattr(init_estimator, 'predict'):
            raise ValueError(
                "The init_estimator should be an object with a predict "
                "function with function signature predict(self, X) "
                "where X is the feature matrix.")
        else:
            self._init_estimator = deepcopy(init_estimator)

        str_kw = {"print_format": "Estimator {}\n{}"}

        super(LucidEnsemble, self).__init__(
            tree_ensemble,
            print_limit=print_limit,
            str_kw=str_kw)

    @property
    def n_estimators(self):
        """ The # of tree-estimators in the ensemble """
        return len(self)

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def feature_names(self):
        return self._feature_names

    def __reduce__(self):
        return (self.__class__, (
            self._seq,
            self.feature_names,
            self._init_estimator,
            self._learning_rate,
            self._print_limit)
        )

    def apply(self, X):
        """ Apply estimators in Tree to a pandas DataFrame.
            The DataFrame columns should be the same as
            the feature_names attribute.
        """
        activated_indices = np.zeros(
            (X.shape[0], self.n_estimators),
            dtype=int)
        for ind, lucid_tree in enumerate(self):
            activated_indices[:, ind] = lucid_tree.apply(X)

        return activated_indices

    def predict(self, X):
        """ Create predictions from a pandas DataFrame.
            The DataFrame columns should be the same as
            the feature_names attribute.
        """
        y_pred = np.zeros(X.shape[0])
        for lucid_tree in self:
            y_pred += lucid_tree.predict(X)

        return y_pred * self.learning_rate + \
            self._init_estimator.predict(X).ravel()

    def compress(self, **kwargs):
        """ Output a CompressedEnsemble object which aggregates all
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
                    + self.learning_rate * leaf_node.value

        return CompressedEnsemble(
            unique_leaves,
            self.feature_names,
            self._init_estimator,
            **kwargs)


class CompressedEnsemble(LeafDictionary):
    """ Compressed Ensemble Tree created from
        LucidEnsemble.compress() method.

        The difference between the two is that LucidEnsemble
        is grouped by estimators while CompressedEnsemble is
        grouped by the leaf paths.
    """

    def __init__(self, tree_leaves, feature_names,
                 init_estimator, print_limit=30):
        """ Construct the LucidTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
            This object should be constructed using LucidEnsemble.compress() method
        """
        if not isinstance(tree_leaves, OrderedDict):
            raise ValueError("An OrderedDict object with keys mapped to LeafPath objects "
                             "should be passed into the constructor. CompressedEnsemble "
                             "methods rely on the order of the passed tree_leaves.")

        if not isinstance(feature_names, Iterable):
            raise ValueError(
                "feature_names should be an iterable object containing the "
                "feature names that the tree was trained on")
        self._feature_names = feature_names
        self._init_estimator = deepcopy(init_estimator)

        super(CompressedEnsemble, self).__init__(
            tree_leaves, print_limit)

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def leaves(self):
        return self.keys()

    @property
    def n_leaves(self):
        return len(self)

    def __reduce__(self):
        return (self.__class__, (
            self._seq, self.feature_names,
            self._init_estimator, self._print_limit)
        )

    def combine_with(self, other, self_weight=0.5, other_weight=0.5):
        """ Combine the CompressedEnsemble with another one

        :param other: a CompressedEnsemble object that this one
            will combine with; the features from other will
            be weighted with weight
        :param weight: the weight for how much of the
            prediction will depend on other. in other words
            new_pred = (1-weight)*self_pred + weight*other_pred
        """
        if not isinstance(other, CompressedEnsemble):
            raise ValueError("The passed in other argument must be "
                             "of type CompressedEnsemble.")

        if self_weight < 0.0 or self_weight > 1.0:
            raise ValueError(' '.join((
                "An invalid self_weight of {} was passed in;".format(self_weight),
                "weight must be in [0.0, 1.0]"))
            )
        if other_weight < 0.0 or other_weight > 1.0:
            raise ValueError(' '.join((
                "An invalid other_weight of {} was passed in;".format(other_weight),
                "weight must be in [0.0, 1.0]"))
            )

        for leaf_path, leaf_value in self.items():
            self[leaf_path] = leaf_value * self_weight
        for leaf_path, leaf_value in other.items():
            if leaf_path in self:
                self[leaf_path] += other_weight * leaf_value
            else:
                self[leaf_path] = other_weight * leaf_value

    def predict(self, X):
        """ Create predictions from a pandas DataFrame.
            The DataFrame should have the same.
        """
        if isinstance(X, DataFrame):
            X = X.values.astype(dtype=np.float64, order='F')

        # if len(self) > 0:
        #     leaf_preds = diags(list(self.values()))

        #     return np.array(
        #         self.compute_activation(X).dot(leaf_preds).sum(axis=1)).ravel() + \
        #         self._init_estimator.predict(X).ravel()

        if len(self) > 0:
            leaf_path, leaf_values = \
                zip(*[(leaf_path, self[leaf_path]) for leaf_path in self.leaves])

            return create_prediction(X, leaf_path, leaf_values) + \
                self._init_estimator.predict(X).ravel()
        else:
            raise ValueError("The CompressedEnsemble has no leaves to create a prediction")

    def compute_activation(self, X, considered_paths=None):
        """ Compute an activation matrix, see below for more details.

        :returns: a scipy sparse csr_matrix with shape (n, m)
            where n is the # of rows for X, m is the # of unique leaves.

            It is a binary matrix with values in {0, 1}.
            A value of 1 in entry row i, column j indicates that leaf is
            activated for datapoint i, leaf j.
        """
        if isinstance(X, DataFrame):
            X = X.values.astype(dtype=np.float64, order='F')

        if considered_paths is not None:
            # if not all(map(lambda x: isinstance(x, LeafPath), considered_paths)):
            #     raise ValueError(
            #         "All elements of considered_paths should be of type LeafPath.")
            filtered_leaves = \
                dict([(path, self[path])
                     for path in considered_paths])
        else:
            filtered_leaves = self

        activation_matrix = lil_matrix(
            (X.shape[0], len(filtered_leaves)), dtype=bool)

        for ind, path in enumerate(filtered_leaves.keys()):
            activated_indices = find_activated(X, path)
            activation_matrix[np.where(activated_indices)[0], ind] = True

        return activation_matrix.tocsr()

    def prune_by_leaves(self, X, y_true, metric_function='mse', n_prunes=None):
        """ Prune out estimators from the ensemble over data
            in X (feature variables) and y (target variable)

        :type X: 2d matrix
        :type y: 1d array-like
        :type metric_function: str
        :type n_prunes: int
        :param X : the feature matrix over which predictions will be made
        :param y : the vector of target variables used to evaluate the score
        :param metric_function : the function that decides the scoring;
            by default, the Rsquared is used
        :param n_prunes: decides the # of prunes to do over the estimators.
            Defaults to None, if None then it will prune until score of the
            ensemble does not degrade from taking out a leaf.
        """
        if n_prunes is None:
            n_prunes = self.n_leaves
        elif n_prunes > self.n_leaves:
            logger.info('The n_prunes passed was larger than the n_leaves '
                        'so n_prunes was set to equal the n_leaves')
            n_prunes = self.n_leaves
        y_true = flatten_1darray(y_true)

        # pred_matrix is such that the sum of row j is
        # the prediction for jth datapoint
        pred_matrix = np.zeros(
            (X.shape[0], self.n_leaves),
            order='F')  # make column-major for optimization
        init_pred = self._init_estimator.predict(X).ravel()
        logger.debug('The pred_matrix has shape {}'
                     .format(pred_matrix.shape))

        leaves = list(self.leaves)
        for ind, path in enumerate(leaves):
            activated_indices = find_activated(X, path)
            pred_matrix[:, ind] = activated_indices * self[path]
        y_pred = np.sum(pred_matrix, axis=1).ravel() + init_pred

        inds_to_prune = find_prune_candidates(
            y_true, y_pred, pred_matrix, metric_function, n_prunes)

        for ind in sorted(inds_to_prune, reverse=True):
            worst_leaf = leaves[ind]
            logger.debug(
                'Deleting leaf {}'.format(worst_leaf))
            self.pop(leaves[ind])
            leaves.pop(ind)

        logger.debug('Finished with {} prunes with {} leaves left'
                     .format(len(inds_to_prune), self.n_leaves))

    def finetune(self, X, y, n_iterations, chunk_size=10, metric_name='mse',
                 zero_bounds=0.001, l2_coef=0.0):
        """ Finetune the leaves

        :param X : the feature matrix over which predictions will be made
        """
        if isinstance(X, DataFrame):
            X = X.values.astype(dtype=np.float64, order='F')
        y_pred = self.predict(X)

        finetune_ensemble(
            self, X, y,
            y_pred, n_iterations=n_iterations,
            chunk_size=chunk_size, metric_name=metric_name,
            zero_bounds=zero_bounds, l2_coef=l2_coef)
