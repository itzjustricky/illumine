"""
    The woodland package is to handle the illumination of
    ensemble models.
"""


from .node_methods import make_LucidSKTree
from .node_methods import make_LucidSKEnsemble
from .node_methods import get_tree_predictions
from .node_methods import unique_leaves_per_sample

from .leaf_analysis import rank_leaves
from .leaf_analysis import aggregate_trained_leaves
from .leaf_analysis import aggregate_tested_leaves
from .leaf_analysis import rank_per_sample

from .leaf_objects import LucidSKEnsemble
from .leaf_objects import LucidSKTree
from .leaf_objects import SKFoliage
from .leaf_objects import SKTreeNode

__all__ = ['node_methods', 'leaf_analysis', 'leaf_objects']
