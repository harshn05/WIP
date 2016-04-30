# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
from __future__ import division

import os
import sys

from mpmath import euler

from evosimcorelibs import libCOLOR as col
import numpy as np


from libc.math cimport sin, cos, tan, acos, atan, abs, sqrt
cimport numpy as np
cimport cython

cdef float pi = np.pi
cdef float tpi = 2 * pi
cdef float pibt = pi / 2
cdef float thpibt = 3 * pi / 2


def Assign_AS_RELATIVE_FRACTION(obj):
    cdef int TEXCount = obj.TEXCount
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int p = obj.p
    cdef float RNUM
    cdef np.ndarray[np.float64_t, ndim = 1] Frac = obj.Frac
    cdef np.ndarray[np.float64_t, ndim = 3] Eul = obj.Eul
    cdef np.ndarray[np.int64_t, ndim = 2] I = obj.I
        
    # AUXILARY TENSORS 
    cdef np.ndarray[np.float64_t, ndim = 3] IC = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] EULER = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] RNUMS = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] line = np.zeros(TEXCount + 1, dtype=np.float64)
    cdef int i, j
    
    
    line[0] = 0 
    for i in range(1, TEXCount + 1):
        line[i] = line[i - 1] + Frac[i - 1]
    
    
    RNUMS = np.random.uniform(0, 1, p)
  
    for i in range(p):
        RNUM = RNUMS[i]
        for j in range(TEXCount):
            if RNUM >= line[j] and RNUM <= line[j + 1]:
                EULER[i, 0] = Eul[j, i, 0]
                EULER[i, 1] = Eul[j, i, 1]
                EULER[i, 2] = Eul[j, i, 2]
                break  

    obj.EULER = EULER
    return obj
            
    
    
