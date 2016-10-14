"""
    Tools to manage the creation of an IPython notebook

    CELLMAP (dict): control the possible cells the user can create
    IPynbCreationManager: Object to handle the creation of an IPython notebook

    @author: Ricky
"""

import codecs
from textwrap import dedent

import IPython.nbformat as nbf
from IPython.nbformat.v4.nbbase import (
    new_code_cell, new_markdown_cell, new_notebook, new_raw_cell
)

from runipy.notebook_runner import NotebookRunner


# control the possible cells the user can create
CELLMAP = dict(
    code=new_code_cell,
    markdown=new_markdown_cell,
    raw=new_raw_cell
)


class CellUnit(object):
    """ Object to store the information of a cell
        before it is created as
    """

    def __init__(self, tag, source, run_flag):
        """
            :param tag (string): specify what sort of cell to create
            :param source (string): the source to be outputted in the cell
            :param run_flag (bool): whether or not to capture the output with the cell
        """
        global CELLMAP
        if tag not in CELLMAP.keys():
            raise ValueError("An invalid tag {} was passed. tag must be \
                              one of the following {}".format(tag, list(CELLMAP.keys())))
        if not isinstance(source, str):
            raise ValueError("source ({}) should be a string".format(source))
        if not isinstance(run_flag, bool):
            raise ValueError("run_flag ({}) should be a boolean".format(run_flag))

        self.__tag = tag
        self.__source = source
        self.__run_flag = run_flag

    @property
    def tag(self):
        return self.__tag

    @property
    def source(self):
        return self.__source

    @property
    def run_flag(self):
        return self.__run_flag

    def get_attributes(self):
        return (self.tag, self.source, self.run_flag)


class IPynbCreationManager(object):
    """ Object to handle the creation of an IPython notebook

        :_cells: a list of cells which will be outputted to the notebook
        :_nb_runner: an object to run the cells
        :_cells_to_run: are the indices of the cells to run
    """

    def __init__(self):
        """ Initialize cells list to be inserted into IPynb and
            NoteBookRunner object to evaluate the source in cells

        """
        self._cells = []
        self._nb_runner = NotebookRunner(nb=None)
        self._cells_to_run = []

    def process_cell(self, cell_unit, **cell_kws):
        """ Create a cell from a section of source

        :param cell_unit: CellUnit object containing information about the cell
        :param cell_kws: keyword arguments to be passed into the new_cell method
        """

        if not isinstance(cell_unit, CellUnit):
            raise ValueError("The cell_unit needs to be an instance of CellUnit object")
        cell_tag, cell_source, run_flag = cell_unit.get_attributes()

        global CELLMAP
        if cell_tag not in CELLMAP.keys():
            raise ValueError("An invalid cell_tag {} was passed. cell_tag must be \
                              one of the following {}".format(cell_tag, list(CELLMAP.keys())))
        cell_source = dedent(cell_source).strip()  # format the cell

        # create cell and run it if run_flag is True
        self._cells.append(CELLMAP[cell_tag](source=cell_source, **cell_kws))
        if run_flag:
            if cell_tag != 'code':
                raise ValueError("Tried to run source on a non-code cell.")
            self._nb_runner.run_cell(self._cells[-1])

    def process_multiple_cells(self, cell_units):
        """ Add a snippet or a cluster of pre-prepared cells.
        """
        for cell_unit in cell_units:
            self.process_cell(cell_unit)

    def save(self, output_path, version, **kwargs):
        """ Save the ipython notebook object into a file

        :param output_path (string): the string of the path of where to output
            the ipython notebook file to
        :param version:
        :param kwargs: the key-word arguments to be passed to the
            IPython.nbformat.write method
        """
        nb_obj = new_notebook(cells=self._cells, metadata={'language': 'python'})

        with codecs.open(output_path, encoding='utf-8', mode='w') as fil:
            nbf.write(nb_obj, fil, version, **kwargs)

        # do some operations on execution here
