from distutils.core import setup
from distutils.extension import Extension
import os, errno, shutil
import platform
from shutil import copyfile
import subprocess
import sys

import Cython.Compiler.Options
from Cython.Distutils import build_ext
import numpy


Cython.Compiler.Options.annotate = True



def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occured


silentremove('__init__.py')
silentremove('__init__.pyc')

mylibs = ['libCONVERT', 'libEVOLVE', 'libMORPH', 'libTEXT', 'libTEXTURIZE', 'libTMATRICES', 'libCOLOR', 'libASSIGN', 'libMETROPOLIS', 'libDUMMY']
# mylibs = ['libTEXTURIZE', 'libVISUALIZE']
p = platform.system()
v = sys.version_info.major



for mylib in mylibs:
    filename = mylib + ".pyx"
   
    if platform.system() == 'Linux':
        ext_module = Extension(
            mylib,
            [filename],
            # extra_compile_args=['-fopenmp', '-O3' ],
            # extra_link_args=['-fopenmp', '-O3'],
            extra_compile_args=['-O3' ],
            extra_link_args=['-O3'],
        )
    
    elif platform.system() == 'Windows':
        ext_module = Extension(
        mylib,
        [filename],
        # extra_compile_args=['-fopenmp', '-O3' ],
        # extra_link_args=['-fopenmp', '-O3'],
        extra_compile_args=['-O3' ],
        extra_link_args=['-O3'],
        
        )
    
    elif platform.system() == 'Darwin':
        ext_module = Extension(
        mylib,
        [filename],
        # extra_compile_args=['-fopenmp', '-O3' ],
        # extra_link_args=['-fopenmp', '-O3'],
        extra_compile_args=['-O3' ],
        extra_link_args=['-O3'],
        
        )
    

    setup(
        cmdclass={'build_ext': build_ext},
        ext_modules=[ext_module],
        include_dirs=[numpy.get_include()]
    )  





open("__init__.py", 'a').close()
