# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
from __future__ import division

from cmath import sqrt

import numpy as np


cimport numpy as np
cimport cython

from libc.math cimport sin, cos, tan, asin, acos, atan, abs, sqrt


def TmatricesTriclinic():
    cdef int k = 0
    cdef np.ndarray[np.float64_t, ndim = 3] T = np.zeros((1, 3, 3), dtype=np.float64)
     
# DEFAULT: NO ROTATION
    T[k, 0, 0] = 1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 1
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = 1  
    return T
    

    
def TmatricesCubic():
    
    cdef int k = 0
    cdef np.ndarray[np.float64_t, ndim = 3] T = np.zeros((24, 3, 3), dtype=np.float64)
    
    
# DEFAULT: NO ROTATION
    T[k, 0, 0] = 1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 1
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = 1  
    k = k + 1   
    
    
# ROTATION BY pi/2, pi, 3pi/2 angles about principle axes: 4 Fold
    T[k, 0, 0] = 1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 0
    T[k, 1, 2] = 1
    T[k, 2, 0] = 0
    T[k, 2, 1] = -1
    T[k, 2, 2] = 0  
    k = k + 1
    
    T[k, 0, 0] = 1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = -1
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = -1
    k = k + 1
    
    T[k, 0, 0] = 1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 0
    T[k, 1, 2] = -1
    T[k, 2, 0] = 0
    T[k, 2, 1] = 1
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = 0
    T[k, 0, 2] = -1
    T[k, 1, 0] = 0
    T[k, 1, 1] = 1
    T[k, 1, 2] = 0
    T[k, 2, 0] = 1
    T[k, 2, 1] = 0
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = -1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 1
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = -1
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = 0
    T[k, 0, 2] = 1
    T[k, 1, 0] = 0
    T[k, 1, 1] = 1
    T[k, 1, 2] = 0
    T[k, 2, 0] = -1
    T[k, 2, 1] = 0
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = 1
    T[k, 0, 2] = 0
    T[k, 1, 0] = -1
    T[k, 1, 1] = 0
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = 1
    k = k + 1
    
    T[k, 0, 0] = -1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = -1
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = 1
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = -1
    T[k, 0, 2] = 0
    T[k, 1, 0] = 1
    T[k, 1, 1] = 0
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = 1
    k = k + 1 
  
# ROTATION BY 2pi/3, 4pi/3 angles about body diagonals: 3 Fold
    T[k, 0, 0] = 0
    T[k, 0, 1] = 1
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 0
    T[k, 1, 2] = 1
    T[k, 2, 0] = 1
    T[k, 2, 1] = 0
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = 0
    T[k, 0, 2] = 1
    T[k, 1, 0] = 1
    T[k, 1, 1] = 0
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 1
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = 0
    T[k, 0, 2] = -1
    T[k, 1, 0] = -1
    T[k, 1, 1] = 0
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 1
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = -1
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 0
    T[k, 1, 2] = 1
    T[k, 2, 0] = -1
    T[k, 2, 1] = 0
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = 0
    T[k, 0, 2] = 1
    T[k, 1, 0] = -1
    T[k, 1, 1] = 0
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = -1
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = -1
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 0
    T[k, 1, 2] = -1
    T[k, 2, 0] = 1
    T[k, 2, 1] = 0
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = 0
    T[k, 0, 2] = -1
    T[k, 1, 0] = 1
    T[k, 1, 1] = 0
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = -1
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = 1
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 0
    T[k, 1, 2] = -1
    T[k, 2, 0] = -1
    T[k, 2, 1] = 0
    T[k, 2, 2] = 0
    k = k + 1
    
# ROTATION BY pi angles about face diagonals: 2 Fold
    T[k, 0, 0] = -1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 0
    T[k, 1, 2] = 1
    T[k, 2, 0] = 0
    T[k, 2, 1] = 1
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = -1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 0
    T[k, 1, 2] = -1
    T[k, 2, 0] = 0
    T[k, 2, 1] = -1
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = 0
    T[k, 0, 2] = 1
    T[k, 1, 0] = 0
    T[k, 1, 1] = -1
    T[k, 1, 2] = 0
    T[k, 2, 0] = 1
    T[k, 2, 1] = 0
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = 0
    T[k, 0, 2] = -1
    T[k, 1, 0] = 0
    T[k, 1, 1] = -1
    T[k, 1, 2] = 0
    T[k, 2, 0] = -1
    T[k, 2, 1] = 0
    T[k, 2, 2] = 0
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = 1
    T[k, 0, 2] = 0
    T[k, 1, 0] = 1
    T[k, 1, 1] = 0
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = -1
    k = k + 1
    
    T[k, 0, 0] = 0
    T[k, 0, 1] = -1
    T[k, 0, 2] = 0
    T[k, 1, 0] = -1
    T[k, 1, 1] = 0
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = -1
    k = k + 1

    return T
    
    
    
    
    
    
def TmatricesHexa():
    
    cdef int k = 0
    cdef float a = 0.5
    cdef float b = sqrt(3.0) / 2
    cdef np.ndarray[np.float64_t, ndim = 3] T = np.zeros((12, 3, 3), dtype=np.float64)
    
    
# DEFAULT: NO ROTATION
    T[k, 0, 0] = 1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 1
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = 1  
    k = k + 1   
    
    
# ROTATION BY pi/3, 2pi/3, pi,4pi/3, 5pi/3 angles about c axis: 6 Fold
    T[k, 0, 0] = a
    T[k, 0, 1] = b
    T[k, 0, 2] = 0
    T[k, 1, 0] = -b
    T[k, 1, 1] = a
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = 1  
    k = k + 1
    
    T[k, 0, 0] = -a
    T[k, 0, 1] = b
    T[k, 0, 2] = 0
    T[k, 1, 0] = -b
    T[k, 1, 1] = -a
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = 1
    k = k + 1
    
    T[k, 0, 0] = -1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = -1
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = 1
    k = k + 1
    
    T[k, 0, 0] = -a
    T[k, 0, 1] = -b
    T[k, 0, 2] = 0
    T[k, 1, 0] = b
    T[k, 1, 1] = -a
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = 1
    k = k + 1
    
    T[k, 0, 0] = a
    T[k, 0, 1] = -b
    T[k, 0, 2] = 0
    T[k, 1, 0] = b
    T[k, 1, 1] = a
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = 1
    k = k + 1
# ROTATION BY ANGLE pi ABOUT BASAL FACE DIAGONALS: 2 Fold
    T[k, 0, 0] = -1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = 1
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = -1
    k = k + 1
    
    T[k, 0, 0] = a
    T[k, 0, 1] = -b
    T[k, 0, 2] = 0
    T[k, 1, 0] = -b
    T[k, 1, 1] = -a
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = -1
    k = k + 1
    
    T[k, 0, 0] = a
    T[k, 0, 1] = b
    T[k, 0, 2] = 0
    T[k, 1, 0] = b
    T[k, 1, 1] = -a
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = -1
    k = k + 1
# ROTATION BY ANGLE pi ABOUNT THE LINE JOINING THE MIDPOINTS OF THE OPPOSITE EDGES ON BASAL PLANE    
    T[k, 0, 0] = 1
    T[k, 0, 1] = 0
    T[k, 0, 2] = 0
    T[k, 1, 0] = 0
    T[k, 1, 1] = -1
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = -1
    k = k + 1 
  
    T[k, 0, 0] = -a
    T[k, 0, 1] = b
    T[k, 0, 2] = 0
    T[k, 1, 0] = b
    T[k, 1, 1] = a
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = -1
    k = k + 1
    
    T[k, 0, 0] = -a
    T[k, 0, 1] = -b
    T[k, 0, 2] = 0
    T[k, 1, 0] = -b
    T[k, 1, 1] = a
    T[k, 1, 2] = 0
    T[k, 2, 0] = 0
    T[k, 2, 1] = 0
    T[k, 2, 2] = -1
    k = k + 1

    return T    
