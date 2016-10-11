"""
    Description:
        Methods for analyzing tree nodes following the Scikit-Learn API

    breakdown_tree : function
    access_data_in_node : function

    Not sure if I want the below functions
    conditional_count : function
    conditional_mean : function

    @author: Ricky
"""

import numpy as np


def breakdown_tree(sk_tree, feature_names=None, display_relation=False, base_adjustment=0):
    """ Breakdown a tree's splits and returns the value of every leaf along with the path of
        splits that led to the leaf

    ..note:
        Scikit-learn represent their trees with nodes (represented by numbers) printed by
        preorder-traversal; number of -2 represents a leaf, the other numbers are by the index
        of the column for the feature

    :param feature_names: names of the features that were used to split the tree
    :param sk_tree: scikit-learn tree object
    :param display_relation: boolean flag, if marked false then only display feature else
        display the relation as well
    :param base_adjustment: shift all the values with a base value
    :returns: tuple of ("path to leaf", leaf value, leaf sample size)
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
    node_ptr, node_path = 0, []  # ptr_stack keeps track of nodes
    for node_ptr in range(n_splits):
        if len(visit_tracker) != 0:
            visit_tracker[-1] += 1  # visiting the child of the latest node

        if features[node_ptr] != -2:  # visiting node
            visit_tracker.append(0)
            if display_relation:
                append_str = "{}<={}".format(feature_names[features[node_ptr]], thresholds[node_ptr])
            else:
                append_str = feature_names[features[node_ptr]]
            node_path.append(append_str)
        else:  # visiting leaf
            all_nodes.append((node_path.copy(), base_adjustment + values[node_ptr][0][0],
                             node_samples[node_ptr]))
            if node_ptr in sk_tree.tree_.children_right:
                # pop out nodes that I am completely done with
                while(len(visit_tracker) > 0 and visit_tracker[-1] == 2):
                    node_path.pop()
                    visit_tracker.pop()
            if (len(node_path) != 0):
                node_path[-1] = node_path[-1].replace("<=", ">")

    return all_nodes


def access_data_in_node(split_strings, data_df):
    """ Access the data that satisfy the conditions of tree splits within a pandas DataFrame
        The pandas DataFrame should contain the feature names inside the split_strings

    :param split_strings: a list of the conditions/splits that characterize a single node
        split_strings should be outputted from breakdown_tree with display_relation=True
    :param data_df: pandas DataFrame that contains data and columns labeled by the feature names
    """
    samples_in_node = data_df.query(' and '.join(split_strings)).copy()
    return samples_in_node


def conditional_count(split_strings_list, data_df):
    """ Count all data that satisfy any of the conditions of tree splits within a DataFrame

    :param split_strings_list: a list of lists of the conditions/splits that characterize the nodes;
        the rules of many nodes are described in split_strings_list;
        each split_strings should be outputted from breakdown_tree with display_relation=True
    :param data_df: pandas DataFrame that contains data and columns labeled by the feature names
    """
    cnt = 0
    for split_strings in split_strings_list:
        tmp_samples = data_df.query(' and '.join(split_strings))
        cnt += tmp_samples.shape[0]
    return cnt


def conditional_mean(node_rules, node_values, data_df):
    """ Access the data that satisfy the conditions of tree splits within a DataFrame

    :param split_strings_list: a list of lists of the conditions/splits that characterize the nodes;
        the rules of many nodes are described in split_strings_list;
        each split_strings should be outputted from breakdown_tree with display_relation=True
    :param data_df: pandas DataFrame that contains data and columns labeled by the feature names;
    :param mean_label: the label of the column to take the mean over
    """
    cnts = []
    for split_strings, val in zip(node_rules, node_values):
        tmp_samples = data_df.query(' and '.join(split_strings))
        cnts.append(tmp_samples.shape[0])
    cnts = np.array(cnts)
    cnts = cnts / np.sum(cnts)

    return np.dot(cnts, np.array(node_values))
