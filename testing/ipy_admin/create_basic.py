"""
    Description:



    @author: Ricky Chang
"""

import json
import os.path
import logging.config
from textwrap import dedent

from illumine.ipy_admin import IPynbCreationManager
from illumine.ipy_admin import CellUnit


def main():
    # Set up logging
    script_dir = os.path.dirname(__file__)
    json_path = "{}/../logging.json".format(script_dir)
    with open(json_path, 'rt') as f:
        config = json.load(f)
    logging.config.dictConfig(config)

    code_snippets = []
    # code_snippets.append("print('hello')")
    # code_snippets.append('foo = 3')
    code_snippets.append(dedent(
        """
            for i in range(10):
                print(i)
        """))

    code_cells = [CellUnit(tag='code', source=x, run_flag=True) for x in code_snippets]
    ipy_manager = IPynbCreationManager()
    ipy_manager.process_multiple_cells(code_cells)
    ipy_manager.save('foobar.ipynb', version=4)

    # def __init__(self, tag, source, run_flag):


# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
