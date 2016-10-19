"""
    Tools to manage the creation of an IPython notebook

    CELLMAP (dict): control the possible cells the user can create
    IPynbCreationManager: Object to handle the creation of an IPython notebook

    TODO:
        * I don't like how I set up iterate_run_cells
        * I can only save as v3 ... conflict of attributes
            - should make own ipython notebook runner only for v4
            - runipy seems to be poorly written

    @author: Ricky
"""

import os
import codecs

from textwrap import dedent

import IPython.nbformat as nbf
from IPython.nbformat.v4.nbbase import (
    new_code_cell, new_markdown_cell, new_notebook, new_raw_cell
)

from runipy.notebook_runner import (NotebookRunner, NotebookError)


# control the possible cells the user can create
CELLMAP = {
    'code': new_code_cell,
    'markdown': new_markdown_cell,
    'raw': new_raw_cell
}


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
            raise ValueError("An invalid tag {} was passed. tag must be one of the following {}"
                             .format(tag, list(CELLMAP.keys())))
        if not isinstance(source, str):
            raise ValueError("source ({}) should be a string".format(source))
        if not isinstance(run_flag, bool):
            raise ValueError("run_flag ({}) should be a boolean".format(run_flag))
        if run_flag is True and tag != 'code':
            raise ValueError("Only code cells should be run, a tag of {} was passed"
                             .format(tag))

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

        Class members:
        :__cells: a list of cells which will be outputted to the notebook
        :__cell_run_flags: a list of booleans determining whether or not
            to run a certain a cell
        :_nb_runner: an object to run the cells
    """

    def __init__(self):
        """ Initialize cells list to be inserted into IPynb and
            NoteBookRunner object to evaluate the source in cells
        """
        self.__cells = []
        self.__cell_run_flags = []
        self._nb_runner = None

    def process_cell(self, cell_unit, cell_kws=dict()):
        """ Create a cell from a section of source. All changes to the state of
            the class should be done through this method or a method calling this method.

        :param cell_unit: CellUnit object containing information about the cell
        :param cell_kws: keyword arguments to be passed into the new_cell method
        """

        if not isinstance(cell_unit, CellUnit):
            raise ValueError("The cell_unit needs to be an instance of CellUnit object")
        cell_tag, cell_source, run_flag = cell_unit.get_attributes()

        global CELLMAP
        if cell_tag not in CELLMAP.keys():
            raise ValueError("An invalid cell_tag {} was passed. cell_tag must be one of the following {}"
                             .format(cell_tag, list(CELLMAP.keys())))
        cell_source = dedent(cell_source).strip()  # format the cell
        self.__cells.append(CELLMAP[cell_tag](source=cell_source, **cell_kws))

        # Only run the code cells should be run
        if run_flag is True and cell_tag != 'code':
            raise ValueError("Tried to run source on a non-code cell.")
        self.__cell_run_flags.append(run_flag)

    def process_multiple_cells(self, cell_units):
        """ Add a cluster of pre-prepared cells """
        for cell_unit in cell_units:
            self.process_cell(cell_unit)

    def iterate_run_cells(self):
        """ Iterate through the cells where the run_flags were set to True """
        if self._nb_runner is None:
            raise AttributeError("Trying to reference NotebookRunner object was referenced before creation")

        for ws in self._nb_runner.nb.worksheets:
            for run_flag, cell in zip(self.__cell_run_flags, ws.cells):
                if run_flag is True:  # more aligned with English language
                    yield cell

    def save(self, output_path, version, skip_exceptions=False, write_kwargs=dict()):
        """ Save the ipython notebook object into a file

        :param output_path (string): the string of the path of where to output
            the ipython notebook file to
        :param version:
        :param skip_exceptions: whether or not to skip_exceptions when running the cells with run_flag set to True
        :param write_kwargs: the key-word arguments to be passed to the
            IPython.nbformat.write method
        """
        nb_obj = new_notebook(cells=self.__cells, metadata={'language': 'python'})
        with codecs.open(output_path, mode='w') as fil:
            nbf.write(nb_obj, fil, version, **write_kwargs)

        # run the cells with run_cells set to True
        if sum(self.__cell_run_flags) != 0:  # if not all run_cells are set to False
            # temporarily change the working directory

            self._nb_runner = NotebookRunner(nb=nbf.read(open(output_path), as_version=3),
                                             working_dir=os.path.dirname(output_path))
            for i, cell in enumerate(self.iterate_run_cells()):
                try:
                    self._nb_runner.run_cell(cell)
                except NotebookError as err:
                    if not skip_exceptions:
                        raise NotebookError(err)
            nbf.write(self._nb_runner.nb,
                      codecs.open(output_path, mode='w'),
                      version=3)
            self._nb_runner.shutdown_kernel()
