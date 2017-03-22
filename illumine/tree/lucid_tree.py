"""


"""

import numpy as np
import pandas as pd

# from collections import Iterable

# from .leaf_dictionary import LeafDictionary
# from .predict_methods import create_apply
# from .predict_methods import create_prediction

# from ._leaf import DecisionPath
# from ._leaf import TreeLeaf

__all__ = ['LucidTree']


class LucidTree(object):
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

    def __init__(self, leaf_table, print_limit=30):
        """ Construct the LucidTree object using a dictionary object indexed
            by the leaf's index in the pre-order traversal of the decision tree.

            The leaf's index is in set [0, k-1] where k is the # of nodes (inner & leaf nodes)

        :param tree_leaves (dict): a dictionary of representations of the leaves from the
            Scikit-learn tree models, the keys are the index of the leaves in the pre-order
            traversal of the decision tree
        :param print_limit (int): configuration for how to print the LucidEnsemble
            out to the console
        """
        self._leaf_table = leaf_table
        # super(LucidTree, self).__init__(
        #     tree_leaves,
        #     print_limit=print_limit)

    def predict(self, X):
        """ Create predictions from a matrix of the feature variables """
        y_pred = np.zeros(X.shape[0])
        # this indicates the trained tree had no splits;
        # possible via building LucidTree from sklearn model
        if len(self) == 1:
            return y_pred
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(dtype=np.float64, order='F')

        return self._leaf_table.predict(X)

    def __len__(self):
        return len(self._leaf_table)

    # def __reduce__(self):
    #     return (self.__class__, (
    #         self._tree_structure,
    #         self._seq,
    #         self._print_limit)
    #     )
