
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('Phd_ext.pyx', language_level="3"))
