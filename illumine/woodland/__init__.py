"""
    The woodland subpackage is for the
    code of ensemble models
"""

from .lucid_ensemble import LucidEnsemble
from .lucid_ensemble import CompressedEnsemble

from .ensemble_factory import make_LucidEnsemble

from .nurture import weighted_nurturing

__all__ = [
    'LucidEnsemble', 'CompressedEnsemble',
    'make_LucidEnsemble', 'weighted_nurturing']
