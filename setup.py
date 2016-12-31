import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "illumine/woodland/_retrieve_leaf_paths.pyx"),
    include_dirs=[numpy.get_include()]
)
