"""
    Utility methods for file-related functionality.

"""

import os
import os.path


def mkdir_p(path):
    """ Credit to tzot in answer to
        http://stackoverflow.com/a/600612/119527
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')
