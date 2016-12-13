"""
    Description:
        Methods for analyzing tree nodes following the Scikit-Learn API

     make_LucidSKTree : factory method for LucidSKTree;
     make_LucidSKEnsemble : factory method for LucidSKEnsemble;
     unique_leaves_per_sample : function
     get_tree_predictions : function

    TODO:
        * if in the future, experience make_LucidSKEnsemble bottleneck use
            multiprocessing library to unravel trees in parallel
        * separate out leaf path retrieval algorithm from make_LucidSKTree
        * add an extra parameter to choose which leaves to consider

    @author: Ricky
"""


from collections import OrderedDict

from .leaf_objects import SKTreeNode
from .leaf_objects import LucidSKTree
from .leaf_objects import LucidSKEnsemble

__all__ = ['make_LucidSKTree', 'make_LucidSKEnsemble']


# TODO: separate the leaf path retrieval algorithm from make_LucidSKTree
def _gather_leaf_paths():
    """ This function is used to gather the paths to all the
        leaves given some of the Scikit-learn attributes
        of a decision tree.
    """
    pass


def make_LucidSKTree(sk_tree, feature_names, float_precision=5,
                     sort_by_index=True, tree_kw=None):
    """ Breakdown a tree's splits and returns the value of every leaf along
        with the path of splits that led to the leaf

    ..note:
        Scikit-learn represent their trees with nodes (represented by numbers)
        printed by preorder-traversal; number of -2 represents a leaf, the other
        numbers are by the index of the column for the feature

    :param feature_names (list): list of names (strings) of the features that
        were used to split the tree
    :param sk_tree: scikit-learn tree object
    :param float_precision (int): to determine what number the node
        values, thresholds are rounded to
    :param tree_kw (dict): key-word arguments to be passed into
        LucidSKTree's constructor

    :returns: LucidSKTree object indexed by their order in the
        pre-order traversal of the Decision Tree
    """
    if tree_kw is None:
        tree_kw = dict()
    elif not isinstance(tree_kw, dict):
        raise ValueError("tree_kw should be of type dict")
    tree_leaves = OrderedDict()

    # values & node_samples are only used in SKTreeNode init
    values = sk_tree.tree_.value
    node_samples = sk_tree.tree_.n_node_samples
    features = sk_tree.tree_.feature
    thresholds = sk_tree.tree_.threshold

    n_splits = len(features)
    if n_splits == 0:
        raise ValueError("The passed tree is empty!")

    # leaf path retrieval algorithm
    tracker_stack = []  # a stack to track if all the children of a node is visited
    leaf_path = []  # ptr_stack keeps track of nodes
    for node_index in range(n_splits):
        if len(tracker_stack) != 0:
            tracker_stack[-1] += 1  # visiting the child of the latest node

        if features[node_index] != -2:  # visiting inner node
            tracker_stack.append(0)
            append_str = "{}<={}".format(
                feature_names[features[node_index]],
                float(round(thresholds[node_index], float_precision)))
            leaf_path.append(append_str)
        else:  # visiting leaf
            tree_leaves[node_index] = \
                SKTreeNode(leaf_path.copy(),
                           float(round(values[node_index][0][0], float_precision)),
                           node_samples[node_index])

            if node_index in sk_tree.tree_.children_right:
                # pop out nodes that I am completely done with
                while(len(tracker_stack) > 0 and tracker_stack[-1] == 2):
                    leaf_path.pop()
                    tracker_stack.pop()
            if len(leaf_path) != 0:
                leaf_path[-1] = leaf_path[-1].replace("<=", ">")
    # end of leaf path retrieval algorithm

    return LucidSKTree(tree_leaves, feature_names, **tree_kw)


def make_LucidSKEnsemble(sk_ensemble, feature_names, init_estimator=None,
                         tree_kw=None, ensemble_kw=None, **kwargs):
    """ Breakdown a tree's splits and returns the value of every leaf along
        with the path of splits that led to the leaf

    ..note:
        Scikit-learn represent their trees with nodes (represented by numbers) printed
        by preorder-traversal; number of -2 represents a leaf, the other numbers are by
        the index of the column for the feature

    :param sk_ensemble: scikit-learn ensemble model object
    :param feature_names (list): list of names (strings) of the features that
        were used to split the tree
    :param init_estimator (function): the initial estimator of the ensemble
        defaults to None, if None then equals Scikit-learn tree's initial estimator
    :param float_precision (int): to determine what number the node
        values, thresholds are rounded to
    :param tree_kw (dict): key-word arguments to be passed into
        LucidSKTree's constructor
    :param ensemble_kw (dict): key-word arguments to be passed into
        LucidSKEnsemble's constructor

    :returns: dictionary of SKTreeNode objects indexed by their order in the
        pre-order traversal of the Decision Tree
    """
    if tree_kw is None:
        tree_kw = dict()
    elif not isinstance(tree_kw, dict):
        raise ValueError("tree_kw should be of type dict")
    if ensemble_kw is None:
        ensemble_kw = dict()
    elif not isinstance(ensemble_kw, dict):
        raise ValueError("tree_kw should be of type dict")

    ensemble_of_leaves = []
    for estimator in sk_ensemble.estimators_:
        estimator = estimator[0]
        ensemble_of_leaves.append(
            make_LucidSKTree(estimator, feature_names, tree_kw=tree_kw, **kwargs))

    if init_estimator is None:
        init_estimator = sk_ensemble._init_decision_function
    elif not callable(init_estimator):
        raise ValueError(
            "The init_estimator should be a callable function that "
            "takes X (feature matrix) as an argument.")

    return LucidSKEnsemble(
        ensemble_of_leaves, feature_names,
        init_estimator=init_estimator,
        learning_rate=sk_ensemble.learning_rate,
        **ensemble_kw)
