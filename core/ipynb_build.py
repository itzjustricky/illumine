"""
    Tools to manage the creation of an IPython notebook

    CellMap (dict): control the possible cells the user can create
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
CellMap = dict(
    code=new_code_cell,
    markdown=new_markdown_cell,
    raw=new_raw_cell
)


class IPynbCreationManager(object):
    """ Object to handle the creation of an IPython notebook """

    def __init__(self):
        """ Initialize cells list to be inserted into IPynb and
            NoteBookRunner object to evaluate the source in cells
        """
        self._cells = []
        self._nb_runner = NotebookRunner(nb=None)

    def create_cell(self, cell_tag, source_piece, run_flag, **cell_kws):
        """ Create a cell from a section of source

        :param cell_tag (string):
        :param source_piece (string): the source to be outputted in the cell
        :param run_flag (bool): whether or not to capture the output with the cell
        :param cell_kws: keyword arguments to be passed into the new_cell method
        """

        if cell_tag not in CellMap.keys():
            raise ValueError("An invalid cell_tag {} was passed. cell_tag must be \
                              one of the following {}".format(cell_tag, list(CellMap.keys())))
        if cell_tag == 'code':
            source_piece = dedent(source_piece)  # Make sure the code source is properly formatted

        self._cells.append(CellMap[cell_tag](source=source_piece, **cell_kws))
        if run_flag:
            if cell_tag != 'code':
                raise ValueError("Tried to run source on a non-code cell.")
            self._nb_runner.run_cell(self._cells[-1])

    def add_snippet(self, snippet_obj):
        """ Add a snippet or a cluster of pre-prepared cells.
        """
        pass

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
