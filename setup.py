# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 13:59:58 2019

@author: sparra
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('Information.pyx', language_level="3"))
