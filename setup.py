import numpy

from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [
    # leaf modules
    Extension(
        language="c",
        name="illumine.tree.leaf_retrieval",
        sources=["illumine/tree/leaf_retrieval.pyx"],
        extra_compile_args=["-O3"]),
    Extension(
        language="c",
        name="illumine.tree.leaf",
        sources=["illumine/tree/leaf.pyx"],
        extra_compile_args=["-O3"]),
    Extension(
        language="c",
        name="illumine.tree.leaftable",
        sources=["illumine/tree/leaftable.pyx"],
        extra_compile_args=["-O3"]),

    # woodland modules
    Extension(
        language="c",
        name="illumine.woodland.compression",
        sources=["illumine/woodland/compression.pyx"],
        extra_compile_args=["-O3"]),
    Extension(
        language="c",
        name="illumine.woodland.leaf_tuning",
        sources=["illumine/woodland/leaf_tuning.pyx"],
        extra_compile_args=["-O3"]),

    # metrics modules
    Extension(
        language="c",
        name="illumine.metrics.score_functions",
        sources=["illumine/metrics/score_functions.pyx"],
        extra_compile_args=["-O3"]),
]

setup(
    name='illumine',
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
