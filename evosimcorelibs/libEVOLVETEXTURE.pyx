# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
from __future__ import division

import math
import os
import random
import sys

import cv2

import numpy as np


from libc.math cimport ceil

cimport numpy as np
cimport cython
# cimport openmp
# from cython cimport boundscheck, wraparound
# from cython.parallel import prange
# from statsmodels.sandbox.distributions.examples.matchdist import low
# from Cython.Shadow import nogil

cdef float pi = np.pi

def plantseed(int myseed):
    if myseed > 0:
        np.random.seed(myseed)
        random.seed(myseed)
        
