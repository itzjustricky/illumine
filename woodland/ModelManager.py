"""
    Description:



    @author: Ricky Chang
"""

from ..utils.file import mkdir_p
from ..core import IPynbCreationManager


class ModelManager(object):
    """ Manage the creation of an IPython Notebook for
        analyzing a model
    """

    def __init__(self, output_dir):
        """
        :param output_dir (string): the directory in which to store
            all the necessary files
        """
        mkdir_p(output_dir)
        self._output_dir = output_dir
        self._ipynb_manager = IPynbCreationManager()

    def add_snippet(self, snippet_obj):
        """ Add a snippet or a cluster of pre-prepared cells.
        """
        pass
