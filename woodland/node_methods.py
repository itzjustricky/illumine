"""
    Description:
        Methods for analyzing tree nodes following the Scikit-Learn API

    breakdown_tree : function
    access_data_in_node : function

    Not sure if I want the below functions
    conditional_count : function
    conditional_mean : function

    TODO:
        * I don't think get_tree_leaves and get_ensemble_leaf_values methods
            are very useful right now ... I need to change them
        * add a function to see which nodes are activated for
            a certain datarow
        * consider using Cython for creating breakdown_tree
            function since there can be a lot of overhead for
            trees with more depth

    @author: Ricky
"""

from collections import OrderedDict
import numpy as np


class SKTreeNode(object):
    """ Object representation a single node of a decision tree """

    def __init__(self, path, value, n_samples, node_index):
        """
        :params path (list): the decision path to the node
        :params value (numeric): the value associated with the node
        :params n_samples (int): the number of samples that reach the node
        :params node_index (int): the index of the node in the pre-order
            traversal of the decision tree;
            node_index in [0, k-1] where k is the # of nodes (inner & leaf)
        """
        self._node_repr = OrderedDict()

        self._node_repr['path'] = path
        self._node_repr['value'] = value
        self._node_repr['n_samples'] = n_samples

        self._node_index = node_index

    def __str__(self):
        node_strings = []
        for key, val in self._node_repr.items():
            node_strings.append("{}: {}".format(key, val))
        return '\n'.join(node_strings)

    def __repr__(self):
        return self.__str__()

    def get_path(self):
        return self._node_repr['path'].__str__()

    def get_value(self):
        return self._node_repr['value']

    def get_n_samples(self):
        return self._node_repr['n_samples']

    def index(self):
        return self._node_index


def breakdown_tree(sk_tree, feature_names=None, display_relation=False,
                   base_adjustment=0, float_precision=3):
    """ Breakdown a tree's splits and returns the value of every leaf along
        with the path of splits that led to the leaf

    ..note:
        Scikit-learn represent their trees with nodes (represented by numbers) printed
        by preorder-traversal; number of -2 represents a leaf, the other numbers are by
        the index of the column for the feature

    :param feature_names (list): list of names (strings) of the features that were used
        to split the tree
    :param sk_tree: scikit-learn tree object
    :param display_relation (bool): if marked false then only display feature else display
        the relation as well; if marked true, the path
    :param base_adjustment (numeric): shift all the values with a base value
    :param float_precision (int): to determine what number the node values, thresholds are
        rounded to

    :returns: list of SKTreeNode objects
    """
    all_nodes = []

    values = sk_tree.tree_.value
    features = sk_tree.tree_.feature
    node_samples = sk_tree.tree_.n_node_samples
    thresholds = sk_tree.tree_.threshold

    n_splits = len(features)
    if n_splits == 0:
        raise ValueError("The passed tree is empty!")

    if feature_names is None:
        feature_names = np.arange(features.max())

    visit_tracker = []  # a stack to track if all the children of a node is visited
    node_index, node_path = 0, []  # ptr_stack keeps track of nodes
    for node_index in range(n_splits):
        if len(visit_tracker) != 0:
            visit_tracker[-1] += 1  # visiting the child of the latest node

        if features[node_index] != -2:  # visiting inner node
            visit_tracker.append(0)
            if display_relation:
                append_str = "{}<={}".format(feature_names[features[node_index]],
                                             round(thresholds[node_index], float_precision))
            else:
                append_str = feature_names[features[node_index]]
            node_path.append(append_str)
        else:  # visiting leaf
            all_nodes.append(
                SKTreeNode(node_path.copy(),
                           base_adjustment + round(values[node_index][0][0], float_precision),
                           node_samples[node_index],
                           node_index))

            if node_index in sk_tree.tree_.children_right:
                # pop out nodes that I am completely done with
                while(len(visit_tracker) > 0 and visit_tracker[-1] == 2):
                    node_path.pop()
                    visit_tracker.pop()
            if (len(node_path) != 0):
                node_path[-1] = node_path[-1].replace("<=", ">")

    return all_nodes


def get_tree_leaves(sk_tree, X):
    """ Retrieve the index, values of all the leaf nodes that
        the decision tree arrives at.

        the index above refers to index of the node in the pre-order
        traversal of the decision tree

    :param sk_tree: scikit-learn tree object
    :param X: array_like or sparse matrix, shape = [n_samples, n_features]
    """
    leaf_indices, leaf_values = np.zeros(X.shape[0]), np.zeros(X.shape[0])

    decision_path = sk_tree.decision_path(X)
    for i, row in enumerate(decision_path):
        leaf_indices[i] = row.nonzero()[1][-1]
        leaf_values[i] = sk_tree.tree_.value[leaf_indices[i]]

    return leaf_indices, leaf_values


def get_ensemble_leaf_values(sk_ensemble, X):
    """ Retrieve the index, values of all the leaf nodes that
        the decision tree arrives at for each decision tree in
        an ensemble

        the index above refers to index of the node in the pre-order
        traversal of the decision tree

    :param sk_tree: scikit-learn tree object
    :param X: array_like or sparse matrix, shape = [n_samples, n_features]
    """

    leaf_indices = np.zeros((X.shape[0], sk_ensemble.n_estimators))
    leaf_values = np.zeros((X.shape[0], sk_ensemble.n_estimators))

    for estimator in sk_ensemble.estimators:
        estimator = estimator[0][0]

        leaf_indices[0, :] = get_tree_leaves(estimator, X)
        leaf_values[0, :] = get_tree_leaves(estimator, X)

    return leaf_indices, leaf_values
