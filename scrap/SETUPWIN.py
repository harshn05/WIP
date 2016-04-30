from distutils.core import setup
import sys, os, zmq

import numpy

from py2exe.build_exe import py2exe
import resourceLIST_rc


sys.setrecursionlimit(5000)

# libzmq.dll is in same directory as zmq's __init__.py

os.environ["PATH"] = \
    os.environ["PATH"] + \
    os.path.pathsep + os.path.split(zmq.__file__)[0]

setup(console=[{"script": "EvoSim.py"}],
       options={ 
           "py2exe": { 
               "includes": 
               ["zmq.utils", "zmq.utils.jsonapi",
                "zmq.utils.strtypes"] } })
