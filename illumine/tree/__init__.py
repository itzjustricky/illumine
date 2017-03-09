"""
    The tree subpackage is for the
    code of tree models
"""

from .lucid_tree import LucidTree

from .tree_factory import make_LucidTree

__all__ = ['LucidTree', 'make_LucidTree']
