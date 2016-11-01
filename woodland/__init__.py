"""
    The woodland package is to handle the illumination of
    ensemble models.
"""

__all__ = ['TreeTrainer', 'IPynbEnsembleManager', 'node_methods', 'snippets']

from .TreeTrainer import TreeTrainer
from .IPynbEnsembleManager import IPynbEnsembleManager

from .snippets import FeatureImportanceSnippet

from .node_methods import unravel_tree
from .node_methods import unravel_ensemble
from .node_methods import get_tree_predictions
from .node_methods import aggregate_trained_leaves
from .node_methods import aggregate_activated_leaves
from .node_methods import unique_leaves_per_sample
from .node_methods import get_top_leaves
# from .node_methods import *
