"""
    Description:


    TODO:


    @author: Ricky Chang
"""

import os
import pickle

from ..core import (BaseSnippet, ModelSnippet)
from ..core import IPynbCreationManager


class IPynbEnsembleManager(object):
    """ Manage the creation of an IPython Notebook for
        analyzing a model
    """

    def __init__(self, sk_model, output_dir, pickle_file):
        """
        :param output_dir (string): the directory in which to store all
            the necessary files
        :param pickle_file (string): the name of the pickle_file to
            output the model into
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("{}/pickles".format(output_dir), exist_ok=True)  # make a directory for pickle files
        pickle.dump(sk_model, open('{}/pickles/{}'.format(output_dir, pickle_file), 'wb'))

        self._pickle_file = pickle_file
        self._output_dir = output_dir
        self._ipynb_manager = IPynbCreationManager()

    @property
    def pickle_file(self):
        return self._pickle_file

    def add_snippet(self, snippet_cls, **kwargs):
        """ Add a snippet or a cluster of pre-prepared cells.

        :param snippet_cls: the class of the snippet, not an instance
        """
        if not issubclass(snippet_cls, BaseSnippet):
            raise ValueError("An invalid snippet class was passed, the snippet class passed"
                             " should be derived from the BaseSnippet.")

        snippet_instance = snippet_cls()
        # if it is a ModelSnippet, then it needs a pickle file
        if issubclass(snippet_cls, ModelSnippet):
            snippet_instance.pickle_file = self._pickle_file
        self._ipynb_manager.process_multiple_cells(snippet_instance.generate_snippet(**kwargs))

    def save(self, notebook_name, version, **kwargs):
        self._ipynb_manager.save("{}/{}".format(self._output_dir, notebook_name),
                                 version, **kwargs)
