"""
    The woodland package is to handle the illumination of
    ensemble models.
"""

__all__ = ['IPynbEnsembleManager', 'node_methods', 'snippets']

from .IPynbEnsembleManager import IPynbEnsembleManager

from .snippets import FeatureImportanceSnippet

from .node_methods import unravel_tree
from .node_methods import unravel_ensemble
from .node_methods import get_tree_predictions
from .node_methods import aggregate_trained_leaves
from .node_methods import aggregate_tested_leaves
from .node_methods import unique_leaves_per_sample
from .node_methods import rank_leaves
from .node_methods import rank_per_sample
# from .node_methods import *
