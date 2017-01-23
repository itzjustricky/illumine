import numpy

from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        language="c",
        name="illumine.woodland._retrieve_leaf_paths",
        sources=["illumine/woodland/_retrieve_leaf_paths.pyx"]),
    Extension(
        language="c",
        name="illumine.core.metrics",
        sources=["illumine/core/metrics.pyx"]),
    Extension(
        language="c",
        name="illumine.woodland.find_prune_candidate",
        sources=["illumine/woodland/find_prune_candidate.pyx"])
]

setup(
    name='illumine',
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
