# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
from __future__ import division

import math
import os
import random
import sys
import time

import cv2
from scipy import integrate

import libMORPH as morph
import numpy as np


from libc.math cimport ceil

cimport numpy as np
cimport cython
# from cython.parallel import prange


cdef float pi = np.pi


def MH1D(myfunc, float xmin, float xmax, int p, type):
    cdef float xold, num, den, A, x1, u, xstar, stdevx
    cdef long long count
    cdef np.ndarray[np.float64_t, ndim = 1] X = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 1] XX = np.zeros(p, dtype=np.int64)

    count = 0

    while True:
        xold = np.random.uniform(xmin, xmax)
        if myfunc(xold) > 0:
            break
    X[count] = xold
    stdevx = xmax - xmin
    while count < p - 1:
        while True:
            xstar = np.random.normal(xold, stdevx)
            if xmin <= xstar <= xmax:
                break
        num = myfunc(xstar)
        den = myfunc(xold)
        A = min(1, num / den)
        x1 = xstar
        u = np.random.rand()
        if u <= A:
            xold = xstar
            count = count + 1
            X[count] = x1
    if type == 1:
        XX = np.array(XX, dtype=np.int64)
        return XX
    else:
        return X

def MH2D(myfunc, float xmin, float xmax, float ymin, float ymax, int p, int type):
    cdef float xold, yold, num, den, A, x1, y1, u, xstar, ystar, stdevx, stdevy
    cdef long long count
    cdef np.ndarray[np.float64_t, ndim = 1] X = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Y = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 1] XX = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] YY = np.zeros(p, dtype=np.int64)
    
       
    count = 0 
    while True:
        xold = np.random.uniform(xmin, xmax)
        yold = np.random.uniform(ymin, ymax)
        if myfunc(xold, yold) > 0:
            break
    X[count] = xold
    Y[count] = yold
    stdevx = xmax - xmin
    stdevy = ymax - ymin
    while count < p - 1:
        while True:
            xstar = np.random.normal(xold, stdevx)
            ystar = np.random.normal(yold, stdevy)
            if xmin <= xstar <= xmax and ymin <= ystar <= ymax:
                break
        num = myfunc(xstar, ystar)
        den = myfunc(xold, yold)
        A = min(1, num / den)
        x1, y1 = xstar, ystar
        u = np.random.rand()
        if u <= A:
            xold, yold = xstar, ystar
            count = count + 1
            X[count] = x1
            Y[count] = y1
            
    if type == 1:  # 1 means int
        XX = np.array(X, dtype=np.int64)
        YY = np.array(Y, dtype=np.int64)
        return XX, YY
    else:  # else means float
        return X, Y



def MH3D(myfunc, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, int p, type):
    cdef float xold, yold, zold, num, den, A, x1, y1, z1, u, xstar, ystar, zstar, stdevx, stdevy, stdevz
    cdef long long count
    cdef np.ndarray[np.float64_t, ndim = 1] X = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Y = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Z = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 1] XX = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] YY = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ZZ = np.zeros(p, dtype=np.int64)

    count = 0
    while True:
        xold = np.random.uniform(xmin, xmax)
        yold = np.random.uniform(ymin, ymax)
        zold = np.random.uniform(zmin, zmax)
        if myfunc(xold, yold, zold) > 0:
            break
    X[count] = xold
    Y[count] = yold
    Z[count] = zold
    stdevx = xmax - xmin
    stdevy = ymax - ymin
    stdevz = zmax - zmin
    while count < p - 1:
        while True:
            xstar = np.random.normal(xold, stdevx)
            ystar = np.random.normal(yold, stdevy)
            zstar = np.random.normal(zold, stdevz)
            if xmin <= xstar <= xmax and ymin <= ystar <= ymax\
                    and zmin <= zstar <= zmax:
                break
        num = myfunc(xstar, ystar, zstar)
        den = myfunc(xold, yold, zold)
        A = min(1, num / den)
        x1, y1, z1 = xstar, ystar, zstar
        u = np.random.rand()
        if u <= A:
            xold, yold, zold = xstar, ystar, zstar
            count = count + 1
            X[count] = x1
            Y[count] = y1
            Z[count] = z1
    if type == 1:
        XX = np.array(X, dtype=np.int64)
        YY = np.array(Y, dtype=np.int64)
        ZZ = np.array(Z, dtype=np.int64)
        return XX, YY, ZZ
    else:
        return X, Y, Z
    
def INVCDF1D(myfunc, float xmin, float xmax, int p):
    cdef int i, j
    cdef float g, u, v1, v2, g1, g2, N, MIN, m, dx, sum
    cdef int num = 1000
    cdef np.ndarray[np.float64_t, ndim = 1] GRID = np.zeros(num, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] VALUE = np.zeros(num, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Myfuncval = np.zeros(num, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] U = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] X = np.zeros(p, dtype=np.float64)
    
    if xmin == xmax:
        return np.ones(p) * xmin 
    
    if xmin > xmax:
        xmin, xmax = xmax, xmin
  
    U = np.random.uniform(0, 1, p)
    GRID = np.linspace(xmin, xmax, num)
    dx = GRID[1] - GRID[0]
    N = integrate.quad(myfunc, xmin, xmax)[0]
    Myfunc = np.vectorize(myfunc)
    Myfuncval = Myfunc(GRID) / N
    
    VALUE[0] = 0
    for i in range(1, num):
        VALUE[i] = VALUE[i - 1] + (Myfuncval[i] + Myfuncval[i - 1]) * 0.5 * dx
        

    for j in range(p):
        u = U[j]
        for i in range(num - 1):
            if u > VALUE[i] and u < VALUE[i + 1]:
                g1 = GRID[i]
                g2 = GRID[i + 1]
                v1 = VALUE[i]
                v2 = VALUE[i + 1]
                m = (v2 - v1) / (g2 - g1)
                X[j] = g1 + (u - v1) / m
                break
    return X
