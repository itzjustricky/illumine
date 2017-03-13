import numpy

from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        language="c",
        name="illumine.tree.leaf_retrieval",
        sources=["illumine/tree/leaf_retrieval.pyx"]),
    Extension(
        language="c",
        name="illumine.tree.predict_methods",
        sources=["illumine/tree/predict_methods.pyx"]),
    Extension(
        language="c",
        name="illumine.metrics.score_functions",
        sources=["illumine/metrics/score_functions.pyx"]),
    Extension(
        language="c",
        name="illumine.woodland.find_prune_candidate",
        sources=["illumine/woodland/find_prune_candidate.pyx"],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp']
    ),
    Extension(
        language="c",
        name="illumine.woodland.leaf_tuning",
        sources=["illumine/woodland/leaf_tuning.pyx"])
]

setup(
    name='illumine',
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
