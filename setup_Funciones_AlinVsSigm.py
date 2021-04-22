from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('Funciones_AlinVsSigm.pyx', language_level="3"))
