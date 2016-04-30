# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
from __future__ import division

import math
import os
import random
import sys
import time

import cv2

import libCOLOR as COLOR
import libCONVERT as conv
import libMETROPOLIS as met
import libMORPH as morph
import numpy as np


from libc.math cimport ceil


cimport numpy as np
cimport cython
# from cython.parallel import prange


cdef float pi = np.pi

def plantseed(int myseed):
    if myseed > 0:
        np.random.seed(myseed)
        random.seed(myseed)

cdef bresint(np.ndarray[np.int_t, ndim=2] I, int xo, int yo, int r):
    cdef long long dum, x, y
    x = r
    y = 0
    dum = 1 - x
    while x >= y:
        I[x + xo, y + yo] = 1
        I[y + xo, x + yo] = 1
        I[-x + xo, y + yo] = 1
        I[-y + xo, x + yo] = 1
        I[-x + xo, -y + yo] = 1
        I[-y + xo, -x + yo] = 1
        I[x + xo, -y + yo] = 1
        I[y + xo, -x + yo] = 1
        y = y + 1
        if dum <= 0:
            dum = dum + 2 * y + 1
        else:
            x = x - 1
            dum = dum + 2 * (y - x) + 1
    return I

cdef void setwindows(int sa):
    if sa == 0:
        cv2.namedWindow("Microstructure Evolution Grey Scale", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Microstructure Evolution Grey Scale")    
    elif sa == 1:
        cv2.namedWindow("Microstructure Evolution Grey Scale", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Microstructure Evolution Random Color", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Microstructure Evolution Random Color")
    elif sa == 2:
        cv2.namedWindow("Microstructure Evolution Grey Scale", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Microstructure Evolution Random Color", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Microstructure Evolution Grey Scale")
    elif sa == 3:
        cv2.namedWindow("Microstructure Evolution Grey Scale", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Microstructure Evolution Random Color", cv2.WINDOW_NORMAL)

cdef void showriteframe(bint sa, bint sf, fd, int r, np.ndarray[np.int64_t, ndim=2] I, np.ndarray[np.double_t, ndim=3] Col , int p, int Iter, int framepause):
    
    if sa == 2:
        cv2.imshow("Microstructure Evolution Random Color", Col)
        cv2.waitKey(framepause)
    elif sa == 0:
        pass
    elif sa == 1:
        cv2.imshow("Microstructure Evolution Grey Scale", I / p)
        cv2.waitKey(framepause)
    elif sa == 3:
        cv2.imshow("Microstructure Evolution Grey Scale", I / p)
        cv2.imshow("Microstructure Evolution Random Color", Col)
        cv2.waitKey(framepause)
  
    if sf is True:
        if sa == 0 or sa == 2 :
            cv2.imwrite(str(fd + "/" + str(Iter) + 'Color_' + '{0:06d}'.format(r) + '.png'), Col * 255)            
        elif sa == 1:
            cv2.imwrite(str(fd + "/" + str(Iter) + 'Grey_' + '{0:06d}'.format(r) + +'.png'), I * 255 / p)            
        elif sa == 3:
            cv2.imwrite(str(fd + "/" + str(Iter) + 'Grey_' + '{0:06d}'.format(r) + '.png'), I * 255 / p)
            cv2.imwrite(str(fd + "/" + str(Iter) + 'Color_' + '{0:06d}'.format(r) + '.png'), Col * 255)
        
cdef long long checkalive(np.ndarray[np.int64_t, ndim=2] I, int m, int n, int Io, int xmin, int xmax, int ymin, int ymax):
    
    cdef long long i , j, io, jo, a
    a = 1
    for i in range(xmin, xmax + 1):
        for j in range(ymin, ymax + 1):
            if I[i, j] == Io:
                io, jo = i + 1, j
                if 0 <= io <= m - 1 and 0 <= jo <= n - 1:
                    if I[io, jo] == 0:
                        return a
                io, jo = i - 1, j
                if 0 <= io <= m - 1 and 0 <= jo <= n - 1:
                    if I[io, jo] == 0:
                        return a
                io, jo = i, j + 1
                if 0 <= io <= m - 1 and 0 <= jo <= n - 1:
                    if I[io, jo] == 0:
                        return a
                io, jo = i, j - 1
                if 0 <= io <= m - 1 and 0 <= jo <= n - 1:
                    if I[io, jo] == 0:
                        return a
    a = 0
    return a

cdef randindex2D(int m, int n, int p):
     
    a = random.sample(xrange(m * n), p)
    X = [0] * p
    Y = [0] * p
    cdef long long i
    for i in range(0, p):
        X[i] = (a[i]) / (n)
        Y[i] = (a[i]) % (n)
    X = np.array(X, dtype=np.int64)
    Y = np.array(Y, dtype=np.int64)
    
    return X, Y



    
def labelsort(long long n, np.ndarray[np.int64_t, ndim=1] X, np.ndarray[np.int64_t, ndim=1] Y):
    # NOTE THAT m is NOT REQUIRED
    cdef long long k
    cdef long long p = X.shape[0]
    cdef np.ndarray[np.int64_t, ndim = 1] XX = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] YY = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] labelsorted = np.zeros(p, dtype=np.int64)
    
    labelsorted = np.sort(n * X + Y)
    for k in range(p):
        XX[k] = labelsorted[k] / n
        YY[k] = labelsorted[k] % n
    return XX, YY


