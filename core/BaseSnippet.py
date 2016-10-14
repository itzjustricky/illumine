"""
    Description:



    @author: Ricky Chang
"""

from abc import abstractmethod
from .ipynb_build import CellUnit


def format_snippet(cell_tuples, run_flag):
    """

    :param cell_tuples (list): a list of tuples of pair (tag, source);
        should be sorted by the order in which they should appear in the
        IPython notebook
    :param run_flag (bool): if True, run all the cells. Note it will take
        longer for program to run
    """
    cell_units = []
    for tag, source in cell_tuples:
        if tag == 'code':
            cell_units.append(CellUnit(tag, source, run_flag))
        else:
            cell_units.append(CellUnit(tag, source, False))

    return cell_units


class BaseSnippet(object):
    """ Base class to represent a snippet object """

    @abstractmethod
    def get_snippet(self):
        pass
