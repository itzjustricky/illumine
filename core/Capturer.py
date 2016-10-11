"""
    Tools to help capture the output of function calls

    I might not need this at all !

    @author: Ricky
"""

import sys
import base64

from io import StringIO
from io import BytesIO


class StdoutCapturer(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


class PlotCapturer(object):

    def __init__(self):
        """ Initialize with a list to store all the base64 code
            representations of plots captured
        """
        self._plots = []

    def capture_pyplot(self, pyplot_module):
        """ Store the image of a matplotlib.pyplot plot

        :returns: base64 code representation of image
        """
        buf = BytesIO()
        pyplot_module.gcf().savefig(buf, format='png')
        buf.seek(0)

        self._plots.append(base64.b64encode(buf.read()))


if __name__ == "__main__":
    from textwrap import dedent
    code_piece = \
        """
        import numpy as np
        print("Hello World!")

        foobar = np.random.rand(30)
        print(foobar)
        """
    code_piece = dedent(code_piece)

    with StdoutCapturer() as captured_output:
        exec(code_piece)

    print("Done!")
