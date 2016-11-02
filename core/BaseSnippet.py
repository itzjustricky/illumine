"""
    Description:



    @author: Ricky Chang
"""

# import six
# from abc import ABCMeta, abstractmethod


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


# TODO: Want to make all BaseSnippet Derived classes have generate_snippet method
# might do through hasattr way
# class BaseSnippet(object, six.with_metaclass(ABCMeta)):
class BaseSnippet(object):
    """ Base class to represent a snippet object """
    pass


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

        # Not sure if defining this function is a good idea
        def fdel(self):
            del self._pickle_file

        return locals()

    pickle_file = property(**pickle_file())
