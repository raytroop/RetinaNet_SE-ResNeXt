import setuptools
from setuptools.extension import Extension
import numpy as np
from Cython.Build import cythonize

extensions = [
    Extension(
        'dataGen.compute_overlap',
        ['dataGen/compute_overlap.pyx'],
        include_dirs=[np.get_include()]
    ),
]

setuptools.setup(
    name='dataGen',
    packages=setuptools.find_packages(),
	# same with `ext_modules=extensions`,
    ext_modules=cythonize(extensions),
    setup_requires=["cython>=0.28"]
)
