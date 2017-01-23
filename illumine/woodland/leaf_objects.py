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

import logging
from copy import deepcopy
from collections import Iterable
from collections import OrderedDict
from functools import total_ordering

import numpy as np
from scipy.sparse import lil_matrix
from pandas import DataFrame

from ..core import LeafDictionary

from .predict_methods import create_prediction
from .predict_methods import create_apply
from .predict_methods import _map_features_to_int
from .predict_methods import _find_activated

from ..utils.array_check import flatten_1darray

__all__ = ['LeafPath', 'SKTreeNode', 'LucidSKTree',
           'LucidSKEnsemble', 'CompressedEnsemble']


@total_ordering
class LeafPath(object):
    """ Object representation of the path to a leaf node """

    def __init__(self, path):
        """ The initializer for LeafPath

        :param path (list): a list of TreeSplit objects which
            is defined in a Cython module.

            A TreeSplit represent a single split in feature data X.
        """
        self._path = path
        self._key = None

    @property
    def path(self):
        return self._path

    def __iter__(self):
        return self.path.__iter__()

    def __str__(self):
        return self.path.__str__()

    def __repr__(self):
        return self.__str__()

    @property
    def key(self):
        """ The key attribute is used for sorting and
            defining the hash of the LeafPath object
        """
        if self._key is None:
            self._key = ''
            # let the key be composed of the feature_names then
            # the numbers this allows more informative sorting
            for split in self.path:
                self._key += split.feature_name
            for split in self.path:
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

    def apply(self, X_df):
        """ Apply trees in Tree to a pandas DataFrame.
            The DataFrame columns should be the same as
            the feature_names attribute.
        """
        activated_indices = np.zeros(X_df.shape[0], dtype=int)
        # this indicates the trained tree had no splits
        # this is possible in Scikit-learn
        if len(self) == 1:
            return activated_indices

        if not isinstance(X_df, DataFrame):
            raise ValueError("Predictions must be done on a Pandas dataframe")
        if not all(X_df.columns == self.feature_names):
            raise ValueError("The passed pandas DataFrame columns should equal the "
                             "contain the feature_names attribute of the LucidSKTree")

        leaf_paths, leaf_indices = \
            zip(*[(leaf.path, leaf_ind) for leaf_ind, leaf in self.items()])

        return create_apply(X_df, leaf_paths, leaf_indices)

    def predict(self, X_df):
        """ Create predictions from a pandas DataFrame.
            The DataFrame columns should be the same as
            the feature_names attribute.
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

        leaf_paths, leaf_values = \
            zip(*[(leaf.path, leaf.value) for leaf in self.values()])

        return create_prediction(X_df, leaf_paths, leaf_values)

    def __reduce__(self):
        return (self.__class__, (
            self._seq,
            self._feature_names,
            self._print_limit)
        )


# Find the worst component of prediction; used in
# LucidSKEnsemble and CompressedEnsemble for pruning
def _find_worst(ensemble_model, y_true, y_pred,
                pred_matrix, score_function):
    """ Used to find the worst column in the pred_matrix

    :param ensemble_model: should be a LucidSKEnsemble or
        CompressedEnsemble object
    :param y_true (np.ndarray): the values of the true target variables
    :param y_pred (np.ndarray): the values of the outputted predicted
        target variabels
    :param pred_matrix (np.ndarray): a matrix of nxm of which each
        column is a component of the prediction
        (i.e. prediction for the jth datapoint is the sum of row j)
    """
    orig_score = ensemble_model.score(y_true, y_pred, score_function)

    logging.getLogger(__name__).debug(
        'The score before pruning is {}'.format(orig_score))

    worst_ind, best_score = 0, -np.inf
    for ind in range(pred_matrix.shape[1]):
        y_pred_tmp = y_pred - pred_matrix[:, ind]

        curr_score = ensemble_model.score(
            y_true=y_true,
            y_pred=y_pred_tmp,
            score_function=score_function)

        if curr_score > best_score:
            logging.getLogger(__name__).debug(
                'The score was updated to {} without column {}.'
                .format(curr_score, ind))
            worst_ind = ind
            best_score = curr_score

    logging.getLogger(__name__).debug(
        'The best-score is {} from pruning'
        .format(best_score, worst_ind))

    if orig_score > best_score:
        return -1
    else:
        return worst_ind


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

    def __init__(self, tree_ensemble, feature_names, init_estimator, loss_function,
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
        self._leaves_count = None
        self._loss = deepcopy(loss_function)

        if not hasattr(init_estimator, 'predict'):
            raise ValueError(
                "The init_estimator should be an object with a predict "
                "function with function signature predict(self, X) "
                "where X is the feature matrix.")
        else:
            self._init_estimator = deepcopy(init_estimator)

        str_kw = {"print_format": "Estimator {}\n{}"}

        super(LucidSKEnsemble, self).__init__(
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
        """ The name of the features that were used
            to train the scikit-learn model
        """
        return self._feature_names

    def __reduce__(self):
        return (self.__class__, (
            self._seq,
            self.feature_names,
            self._init_estimator,
            self._loss,
            self._learning_rate,
            self._print_limit)
        )

    def apply(self, X_df):
        """ Apply estimators in Tree to a pandas DataFrame.
            The DataFrame columns should be the same as
            the feature_names attribute.
        """
        activated_indices = np.zeros(
            (X_df.shape[0], self.n_estimators),
            dtype=int)
        for ind, lucid_tree in enumerate(self):
            activated_indices[:, ind] = lucid_tree.apply(X_df)

        return activated_indices

    def score(self, y_true, y_pred, score_function=None):
        y_true = flatten_1darray(y_true)
        if score_function is None:
            return -1.0 * self._loss(y_true, y_pred)
        else:
            return score_function(y_true, y_pred)

    def predict(self, X_df):
        """ Create predictions from a pandas DataFrame.
            The DataFrame columns should be the same as
            the feature_names attribute.
        """
        y_pred = np.zeros(X_df.shape[0])
        for lucid_tree in self:
            y_pred += lucid_tree.predict(X_df)

        return y_pred * self.learning_rate + \
            self._init_estimator.predict(X_df).ravel()

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
            self._loss,
            **kwargs)

    def prune_by_estimators(self, X_df, y_true,
                            score_function=None, n_prunes=None):
        """ Prune out estimators from the ensemble over data
            in X_df (feature variables) and y (target variable)

        :param X_df (pandas.DataFrame): the feature matrix over
            which predictions will be made
        :param y_true (1d vector): the vector of target variables
            used to evaluate the score
        :param score_function (function): the function that decides
            the scoring; by default, the negative loss function is used
        :param n_prunes: decides the # of prunes to do over the
            estimators. Defaults to None, if None then it will
            prune until score of the ensemble does not degrade
            from taking out an estimator
        """
        if n_prunes is None:
            n_prunes = self.n_estimators
        y_true = flatten_1darray(y_true)

        # pred_matrix is such that the sum of row j is
        # the prediction for jth datapoint
        pred_matrix = np.zeros((X_df.shape[0], self.n_estimators))
        init_pred = self._init_estimator.predict(X_df).ravel()
        logging.getLogger(__name__).debug(
            'The pred_matrix has shape {}'.format(pred_matrix.shape))

        for est_ind, lucid_tree in enumerate(self):
            pred_matrix[:, est_ind] = lucid_tree.predict(X_df)
        pred_matrix *= self.learning_rate
        y_pred = np.sum(pred_matrix, axis=1).ravel() + init_pred

        # prune phase
        for prune_ind in range(n_prunes):
            worst_est_ind = _find_worst(
                self, y_true, y_pred,
                pred_matrix, score_function)
            if worst_est_ind == -1:  # if no score improvement
                break
            # updates to object state after finding worst estimator
            logging.getLogger(__name__).debug(
                'Deleting estimator {} in the {} prune'
                .format(self[worst_est_ind], prune_ind))
            self.pop(worst_est_ind)

            # update to prediction value
            y_pred = y_pred - pred_matrix[:, worst_est_ind]
            pred_matrix = np.delete(pred_matrix, worst_est_ind, axis=1)

        logging.getLogger(__name__).info(
            'Finished with {} prunes with {} estimators left'
            .format(prune_ind))


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
            raise ValueError("The passed tree_leaves needs to be a dict object")
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

    def __init__(self, tree_leaves, feature_names, init_estimator,
                 loss_function, print_limit=30):
        """ Construct the LucidSKTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)
            This object should be constructed using LucidSKEnsemble.compress() method
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
        self._loss = loss_function

        super(CompressedEnsemble, self).__init__(
            tree_leaves, print_limit)

    @property
    def feature_names(self):
        """ The name of the features that were used
            to train the scikit-learn model
        """
        return self._feature_names

    @property
    def leaves(self):
        return self.keys()

    @property
    def n_leaves(self):
        return len(self)

    def __reduce__(self):
        return (self.__class__, (
            self._seq,
            self.feature_names,
            self._init_estimator,
            self._loss,
            self._print_limit)
        )

    def score(self, y_true, y_pred, score_function=None):
        y_true = flatten_1darray(y_true)
        if score_function is None:
            return -1.0 * self._loss(y_true, y_pred)
        else:
            return score_function(y_true, y_pred)

    def predict(self, X_df):
        """ Create predictions from a pandas DataFrame.
            The DataFrame should have the same.
        """
        if not isinstance(X_df, DataFrame):
            raise ValueError("Predictions must be done on a Pandas dataframe")
        if not all(X_df.columns == self.feature_names):
            raise ValueError("The passed dataframe should")

        leaf_path, leaf_values = \
            zip(*[(leaf_path, self[leaf_path]) for leaf_path in self.leaves])

        return create_prediction(X_df, leaf_path, leaf_values) + \
            self._init_estimator.predict(X_df).ravel()

    def compute_activation(self, X_df, considered_paths=None):
        """ Compute an activation matrix, see below for more details.

        :returns: a scipy sparse csr_matrix with shape (n, m)
            where n is the # of rows for X_df, m is the # of unique leaves.

            It is a binary matrix with values in {0, 1}.
            A value of 1 in entry row i, column j indicates that leaf is
            activated for datapoint i, leaf j.
        """
        if not isinstance(X_df, DataFrame):
            raise ValueError("Predictions must be done on a Pandas dataframe")
        if not all(X_df.columns == self.feature_names):
            raise ValueError("The passed pandas DataFrame columns should equal the "
                             "contain the feature_names attribute of the LucidSKTree")

        f_map = _map_features_to_int(X_df.columns)
        X = X_df.values

        if considered_paths is not None:
            if not all(map(lambda x: isinstance(x, LeafPath), considered_paths)):
                raise ValueError(
                    "All elements of considered_paths should be of type LeafPath.")
            filtered_leaves = \
                dict([(path, self[path])
                     for path in considered_paths])
        else:
            filtered_leaves = self

        activation_matrix = lil_matrix(
            (X_df.shape[0], len(filtered_leaves)),
            dtype=bool)
        # Make sure the activation_matrix is initialized
        # with False boolean values
        assert(activation_matrix.sum() == 0)

        for ind, path in enumerate(filtered_leaves.keys()):
            activated_indices = _find_activated(X, f_map, path)
            activation_matrix[np.where(activated_indices)[0], ind] = True

        return activation_matrix.tocsr()

    def prune_by_leaves(self, X_df, y_true,
                        score_function=None, n_prunes=None):
        """ Prune out estimators from the ensemble over data
            in X_df (feature variables) and y (target variable)

        :param X_df (pandas.DataFrame): the feature matrix over
            which predictions will be made
        :param y (1d vector): the vector of target variables
            used to evaluate the score
        :param score_function (function): the function that decides
            the scoring; by default, the negative loss function is used
        :param n_prunes: decides the # of prunes to do over the
            estimators. Defaults to None, if None then it will
            prune until score of the ensemble does not degrade
            from taking out an estimator
        """
        if n_prunes is None:
            n_prunes = self.n_leaves
        y_true = flatten_1darray(y_true)  # check y_true is flat

        # pred_matrix is such that the sum of row j is
        # the prediction for jth datapoint
        pred_matrix = np.zeros((X_df.shape[0], self.n_leaves))
        init_pred = self._init_estimator.predict(X_df).ravel()
        logging.getLogger(__name__).debug(
            'The pred_matrix has shape {}'.format(pred_matrix.shape))

        f_map = _map_features_to_int(X_df.columns)
        X = X_df.values

        leaves = list(self.leaves)
        for ind, path in enumerate(leaves):
            activated_indices = _find_activated(X, f_map, path)
            pred_matrix[:, ind] = activated_indices * self[path]

        y_pred = np.sum(pred_matrix, axis=1).ravel() + init_pred
        # prune phase
        for prune_ind in range(n_prunes):
            worst_leaf_ind = _find_worst(
                self, y_true, y_pred,
                pred_matrix, score_function)
            if worst_leaf_ind == -1:  # if no score improvement
                break
            # updates to object state after finding worst leaf
            worst_leaf = leaves[worst_leaf_ind]
            logging.getLogger(__name__).debug(
                'Deleting leaf {} in the {} prune'.format(worst_leaf, prune_ind))
            self.pop(leaves[worst_leaf_ind])
            leaves.pop(worst_leaf_ind)

            # update to prediction value
            y_pred = y_pred - pred_matrix[:, worst_leaf_ind]
            pred_matrix = np.delete(pred_matrix, worst_leaf_ind, axis=1)

        logging.getLogger(__name__).debug(
            'Finished with {} prunes with {} leaves left'
            .format(prune_ind, self.n_leaves))
