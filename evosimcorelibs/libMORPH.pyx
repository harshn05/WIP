# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
from __future__ import division

from cmath import sqrt
from math import gamma
import math
import os
import random
import sys

import cv2

from cython.parallel import prange
import libCONVERT as conv
import numpy as np


from libc.math cimport ceil

cimport numpy as np
cimport cython


from libc.math cimport sin, cos, tan, asin, acos, atan, abs, sqrt

cdef float pibt = np.pi / 2
cdef float tpibth = 2 * np.pi / 3
cdef float fpibth = 4 * np.pi / 3
cdef float pi = np.pi
cdef float thpibt = 3 * np.pi / 2
cdef float tpi = 2 * np.pi


def areadistribution(np.ndarray[np.int64_t, ndim=2] I):
    cdef long long i, j, p, m, n, label    
    m = I.shape[0]
    n = I.shape[1]
    p = np.max(I)
    cdef np.ndarray[np.int64_t, ndim = 1] A = np.zeros(p, dtype=np.int64)
    
    for i in range(m):
        for j in range(n):
            label = I[i, j]
            A[label - 1] = A[label - 1] + 1
            
    return A

def triplepoints(np.ndarray[np.int64_t, ndim=2] I, int m, int n, long long p):
    
    cdef int i, j, count, M1, M2, M3, M4
    cdef long long maxtrip = p ** 2
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(maxtrip, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(maxtrip, dtype=np.int64)
       
    count = 0
    for i in range(m - 1):
        for j in range(n - 1):
            M1 = I[i, j]
            M2 = I[i, j + 1]
            M3 = I[i + 1, j + 1]
            M4 = I[i + 1, j]
            
            if M1 == M2 and M1 != M3 and M2 != M4 and M3 != M4:
                X[count] = i
                Y[count] = j
                count = count + 1
            elif M2 == M3 and M2 != M4 and M3 != M1 and M4 != M1:
                X[count] = i
                Y[count] = j
                count = count + 1
            elif M3 == M4 and M3 != M1 and M4 != M2 and M1 != M2:
                X[count] = i
                Y[count] = j
                count = count + 1
            elif M4 == M1 and M4 != M2 and M1 != M3 and M2 != M3:
                X[count] = i
                Y[count] = j
                count = count + 1

    cdef np.ndarray[np.int64_t, ndim = 1] XX = np.zeros(count, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] YY = np.zeros(count, dtype=np.int64)
    
    
    for i in range(count):
        XX[i] = X[i]
        YY[i] = Y[i]
    return XX + 0.5, YY + 0.5
            
            
            
            
            
    
    
def centroids(np.ndarray[np.int64_t, ndim=2] I, int m, int n, int p):
    cdef long i, j, label, count    
    cdef np.ndarray[np.float64_t, ndim = 1] Xo = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Yo = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 1] Xc = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Yc = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] Count = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 1] Sumx = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Sumy = np.zeros(p, dtype=np.int64)
    
    for i in range(m):
        for j in range(n):
            label = I[i, j]
            Count[label - 1] = Count[label - 1] + 1 
            Sumx[label - 1] = Sumx[label - 1] + i
            Sumy[label - 1] = Sumy[label - 1] + j
    
    for label in range(p):
        Xo[label] = Sumx[label] / Count[label]
        Yo[label] = Sumy[label] / Count[label]
    
    XCC = np.round(Xo)
    YCC = np.round(Yo)
    Xc = np.array(XCC, dtype=np.int64)
    Yc = np.array(YCC, dtype=np.int64)
    return Xc, Yc



def segment(np.ndarray[np.int64_t, ndim=2] I):
    cdef long long  m, n, p, i, j, l     
    m = I.shape[0]
    n = I.shape[1]
    p = np.max(I)

    cdef np.ndarray[np.int64_t, ndim = 3] J = np.zeros((p, m, n), dtype=np.int64)
    
    for i in range(m):
        for j in range(n):
            l = I[i, j] - 1
            J[l, i, j] = 1
    return J

def neighbors(np.ndarray[np.int64_t, ndim=2] I):
    cdef long long  m, n, p, i, j, l1, l2, pairs, low, high, go, count
    cdef float checkme
    m = I.shape[0]
    n = I.shape[1]
    p = np.max(I)
    pairs = p * (p + 1) / 2
    cdef np.ndarray[np.int64_t, ndim = 1] Nei = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] Pairs = np.zeros(pairs, dtype=np.float64)
    
    count = 0
    for i in range(m):
        for j in range(n - 1):
            l1 = I[i, j]
            l2 = I[i, j + 1]
            if(l1 != l2):
                if l1 < l2:
                    low = l1
                    high = l2
                else:
                    low = l2
                    high = l1
                
                go = 1
                checkme = 0.5 * (low + high) * (low + high + 1) + low
                for k in range(count):
                    if(Pairs[k] == checkme):
                        go = 0
                        break
                if go == 1:   
                    Nei[l1 - 1] = Nei[l1 - 1] + 1 
                    Nei[l2 - 1] = Nei[l2 - 1] + 1
                    Pairs[count] = checkme
                    count = count + 1
    
    for j in range(n):
        for i in range(m - 1):
            l1 = I[i, j]
            l2 = I[i + 1, j]
            if(l1 != l2):
                if l1 < l2:
                    low = l1
                    high = l2
                else:
                    low = l2
                    high = l1
                
                go = 1
                checkme = 0.5 * (low + high) * (low + high + 1) + low
                for k in range(count):
                    if(Pairs[k] == checkme):
                        go = 0
                        break
                if go == 1:   
                    Nei[l1 - 1] = Nei[l1 - 1] + 1 
                    Nei[l2 - 1] = Nei[l2 - 1] + 1
                    Pairs[count] = checkme
                    count = count + 1
    
    return Nei

                
                 
                
            
    






        
        
    
    
    
