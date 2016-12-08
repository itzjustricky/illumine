"""
    Description:



    @author: Ricky Chang
"""

import six
from ..util.class_tools import AccessMeta


# I should move this away from this module
def format_snippet(cell_tuples, run_flag):
    """

    :param cell_tuples (list): a list of tuples of pair (tag, source);
        should be sorted by the order in which they should appear in the
        IPython notebook
    :param run_flag (bool): if True, run all the cells. Note it will take
        longer for program to run
    """
    from .ipynb_build import CellUnit

    cell_units = []
    for tag, source in cell_tuples:
        if tag == 'code':
            cell_units.append(CellUnit(tag, source, run_flag))
        else:
            cell_units.append(CellUnit(tag, source, False))

    return cell_units


class BaseSnippet(object, six.with_metaclass(AccessMeta)):
    """ Base class to represent a snippet object.
        The derived classes should not override the __init__ function.
    """

    # The __init__ function here makes sure the generate_snippet is defined
    @AccessMeta.final
    def __init__(self):
        # check to see Snippet has a generate_snippet function
        if not hasattr(self, "generate_snippet"):
            raise AttributeError("Derived classes of BaseSnippet must have the "
                                 "generate_snippet method defined.")
            # check it is callable
            if not getattr(self, "generate_snippet"):
                raise AttributeError("generate_snippet should be callable")

    @classmethod
    def display_info(cls):
        """ Print out the parameters needed for the generate_snippet function """
        print(cls.generate_snippet.__doc__)


class ModelSnippet(BaseSnippet):
    """ Base class to represent a snippet object for a model;
        The main difference is a ModelSnippet requires a model and
        a path to access the pickle file containing the model.
    """

    def pickle_file():
        doc = " The pickle_file property "

        def fget(self):
            """ The get function for retrieving pickle_file property """
            try:
                return self._pickle_file
            except AttributeError:
                raise AttributeError("You must set a pickle file for a ModelSnippet!")

        def fset(self, file_name):
            """ The function that defines how the pickle_file should be set """
            if isinstance(file_name, str):
                self._pickle_file = file_name
            else:
                raise ValueError("The pickle_file must be set as a string type.")

        return locals()

    pickle_file = property(**pickle_file())
