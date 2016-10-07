"""
    Description:


    TODO:
        * should a base class be made so that a separate breakdown is
          made for each class?

    @author: Ricky Chang
"""

import os
import os.path
import pickle

from abc import abstractmethod

from IPython.nbformat.v4.nbbase import (
    new_code_cell, new_markdown_cell, new_notebook,
    new_output, new_raw_cell
)

from ...utils import file_utils

"""
from base64 import encodestring
from IPython.nbformat.v4.nbbase import (
    new_code_cell, new_markdown_cell, new_notebook,
    new_output, new_raw_cell
)
"""


class BreakdownGenerator(object):

    def __init__(self, report_dir, tree_model):
        pass

    @abstractmethod
    def generate_nb(self, filename):
        pass


class GradientBoostingGenerator(BreakdownGenerator):

    """
    # change to control what features to output
    def __init__(self, model_obj=None, pickle_dir=None, pickle_path=None):
        if pickle_path is None:
            raise ValueError("GradientBoostingGenerator must be passed either a model_obj or pickle_path parameter.")

        if pickle_path is None:
            pickle.dump(model_obj, open('{}/{}.pkl'.format(report_dir, model_name), 'wb'))
        self._pickle_path = pickle_path
        # self._model_name = model_name
    """

    @staticmethod
    def output_pickle(model_obj, pickle_path):
        """ Output a pickle file of the model object.
            Creates any non-existent directories leading to the path of the pickle_path

        :returns: TODO
        """
        pass

    def generate_nb(self, model_name, pickle_path):
        if not os.path.isfile(pickle_path):
            raise ValueError("The pickle_path does not lead to a file")

        cells = []
        # Set up the ipython notebook to have the ipython object
        load_model_source = \
            """
                {} = pickle.load(open('{}', 'rb'))
            """.format(model_name, pickle_path)
        cells.append(new_code_cell(
            source=load_model_source,
            execution_count=None,
        ))

        out_dir = os.path.dirname(os.path.abspath(pickle_path))
        pass
