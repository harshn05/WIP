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
cdef float tpi = 2 * pi
cdef float pibt = pi / 2
cdef float thpibt = 3 * pi / 2

def EULToCol(np.ndarray[np.float64_t, ndim=3] EUL):
    cdef int i, j, m, n, label
    m = EUL.shape[0]
    n = EUL.shape[1]
    cdef np.ndarray[np.float64_t, ndim = 3] COL = np.zeros((m, n, 3), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            COL[i, j, 2] = EUL[i, j, 0] / tpi
            COL[i, j, 1] = EUL[i, j, 1] / pi
            COL[i, j, 0] = EUL[i, j, 2] / tpi
            
    return COL

def EulToCol(np.ndarray[np.int64_t, ndim=2] I, np.ndarray[np.float64_t, ndim=2] Eul, int p):
    cdef int i, j, m, n, label
    m = I.shape[0]
    n = I.shape[1]
    cdef np.ndarray[np.float64_t, ndim = 2] M = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    
        
    for i in range(p):
        M[i, 0] = Eul[i, 0] / tpi;
        M[i, 1] = Eul[i, 1] / pi;
        M[i, 2] = Eul[i, 2] / tpi;
        
    
    for i in range(m):
        for j in range(n):
            label = I[i, j]
            Col[i, j, 2] = M[label - 1, 0]
            Col[i, j, 1] = M[label - 1, 1]
            Col[i, j, 0] = M[label - 1, 2]
            
    return Col


