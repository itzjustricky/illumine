"""
    The woodland package is to handle the illumination of
    ensemble models.
"""

__all__ = ['TreeTrainer', 'IPynbEnsembleManager', 'node_methods', 'snippets']

from .TreeTrainer import TreeTrainer
from .IPynbEnsembleManager import IPynbEnsembleManager

from .snippets import FeatureImportanceSnippet
from .node_methods import *
