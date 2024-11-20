import os
from Cython.Build import cythonize
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import sys


"""
For some reason, the libraries extension does not support extra compiler arguments. So
instead, we pass them directly by changing the evironment variable CFLAGS
"""
compiler_flags = ["-std=c++11", "-fopenmp", "-march=native"]
libraries = []

# set some flags and libraries depending on the system platform
if sys.platform.startswith('win32'):
    compiler_flags.append('/Od')
    libraries.append('gomp')
elif sys.platform.startswith('darwin'):
    compiler_flags.append('-O2')
    libraries.append('omp')
elif sys.platform.startswith('linux'):
    compiler_flags.append('-O2')
    libraries.append('gomp')
    

CFLAGS = os.getenv("CFLAGS")
if CFLAGS is None:
    CFLAGS = ""
else:
    CFLAGS = str(CFLAGS)
os.environ["CFLAGS"] = CFLAGS
os.environ["CFLAGS"] += " "
os.environ["CFLAGS"] += ' '.join(compiler_flags)
base_path = sys.prefix

full_dependence = ["cpp/src/spline.cpp"]

cpu_extension = dict(
    libraries=libraries,
    language='c++',
    include_dirs=["cpp/include", np.get_include(), base_path + "/include"],
)

wave_ext = Extension(
    "splinecy", 
    sources=["cython/spline_wrap.pyx", *set(full_dependence)], 
    **cpu_extension,
)

ext_modules = [wave_ext]

setup(
    name = "multispline",
    authors = "Zach Nasipak",
    version = "0.1.0",
    description = "Cubic splines in multiple dimensions",
    ext_modules = cythonize(ext_modules, language_level = "3"),
    py_modules = ["multispline.spline"],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Natural Language :: English",
        "Programming Language :: C++",
        "Programming Language :: Cython",
    ],
    cmdclass = {'build_ext': build_ext},
    zip_safe = False
)