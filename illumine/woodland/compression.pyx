"""
    Module to contain code to aggregate leaves
    with the same leaf_hash and their values

"""

from ..tree.leaftable cimport LeafTable
from ..tree.leaf cimport TreeLeaf


cpdef compress_leaves(list leaf_tables, double weight):
    """ Aggregate all the leaves in the leaf_tables
        into one leaf_table
    """
    cdef dict leaves_dict = dict()

    cdef LeafTable leaf_table
    cdef TreeLeaf tree_leaf
    # iterate over all tree leaves and store them in a dictionary
    for leaf_table in leaf_tables:
        for tree_leaf in leaf_table:
            leaves_dict.setdefault(tree_leaf.leaf_hash, []) \
                .append(tree_leaf)

    # aggregated value over leaves with the same leaf_hash
    # all weighted by the passed weight argument
    cdef double aggr_value
    cdef list tree_leaves, aggr_tree_leaves
    aggr_tree_leaves = []
    # aggregate all the tree_leaves with the same hash
    for key, tree_leaves in leaves_dict.items():
        aggr_value = 0.0
        for tree_leaf in tree_leaves:
            aggr_value += weight * tree_leaf.value

        # create the new "aggregated" TreeLeaf
        aggr_tree_leaves.append(
            TreeLeaf(tree_leaf.tree_splits, aggr_value))

    return LeafTable(aggr_tree_leaves)
