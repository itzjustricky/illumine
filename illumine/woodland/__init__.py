"""
    The woodland package is to handle the illumination of
    ensemble models.
"""


from .factory_methods import make_LucidSKTree
from .factory_methods import make_LucidSKEnsemble
from .factory_methods import get_tree_predictions
from .factory_methods import unique_leaves_per_sample

from .leaf_analysis import rank_leaves
from .leaf_analysis import aggregate_trained_leaves
from .leaf_analysis import aggregate_tested_leaves
from .leaf_analysis import rank_per_sample

from .leaf_objects import LucidSKEnsemble
from .leaf_objects import LucidSKTree
from .leaf_objects import SKFoliage
from .leaf_objects import SKTreeNode

__all__ = ['factory_methods', 'leaf_analysis', 'leaf_objects']
