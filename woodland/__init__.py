"""
    The woodland package is to handle the illumination of
    ensemble models.
"""

__all__ = ['TreeTrainer', 'IPynbEnsembleManager', 'feature_importance', 'node_methods']

from .TreeTrainer import TreeTrainer
from .IPynbEnsembleManager import IPynbEnsembleManager
from .feature_importance import FeatureImportanceSnippet

from .node_methods import *
