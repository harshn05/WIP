# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
from __future__ import division

import time

from numpy import arccos

from cython.parallel import prange
import libCONVERT as conv
import libMORPH as morph
import libTMATRICES as TMAT
import numpy as np


cimport numpy as np
cimport cython


from libc.math cimport sin, cos, tan, asin, acos, atan, abs, sqrt

cimport openmp

cdef float pibt = np.pi / 2
cdef float tpibth = 2 * np.pi / 3
cdef float fpibth = 4 * np.pi / 3
cdef float pi = np.pi
cdef float thpibt = 3 * np.pi / 2
cdef float tpi = 2 * np.pi



def disorientation(np.ndarray[np.int64_t, ndim=2] I, np.ndarray[np.float64_t, ndim=3] Rot, int sym , int type):
    if sym == 1:
        theta = disorientationTRICLINIC(I, Rot, type)
    else:
        theta = disorientationSYM(I, Rot, sym, type)
    
    return theta

def disorientationTRICLINIC(np.ndarray[np.int64_t, ndim=2] I, np.ndarray[np.float64_t, ndim=3] Rot, int type):
    cdef int ii, jj, i, j, p, m, n, count, l1, l2, times, f
    cdef float sum, mycos, sum2, thetaold, thetanew    
    p = Rot.shape[0]
    m = I.shape[0]
    n = I.shape[1]
    times = p * (p - 1) / 2
    cdef np.ndarray[np.float64_t, ndim = 1] theta = np.zeros(times, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 2] Freq = np.zeros((p, p), dtype=np.int64)
 
    if type == 1: 
        count = 0
        for i in range(m):
            for j in range(n - 1):
                l1 = I[i, j] - 1
                l2 = I[i, j + 1] - 1
                if(l1 != l2):
                    if l1 < l2:
                        low = l1
                        high = l2
                    else:
                        low = l2
                        high = l1
                    Freq[low, high] = Freq[low, high] + 1
                    count = count + 1
        
        
        for j in range(n):
            for i in range(m - 1):
                l1 = I[i, j] - 1
                l2 = I[i + 1, j] - 1
                if(l1 != l2):
                    if l1 < l2:
                        low = l1
                        high = l2
                    else:
                        low = l2
                        high = l1
                    Freq[low, high] = Freq[low, high] + 1
                    count = count + 1
      
    else:
        for ii in range(p - 1):
            for jj in range(ii + 1, p):
                Freq[ii, jj] = 1
    

    count = 0
    for ii in range(p - 1):
        for jj in range(ii + 1, p):
            f = Freq[ii, jj]
            if (f > 0):
                sum = 0.0
                for i in range(3):
                    for j in range(3):
                        sum = sum + Rot[ii, i, j] * Rot[jj, i, j]
                mycos = (sum - 1) / 2
                
                if mycos >= -1 and mycos <= 1:
                    for i in range(count, count + f + 1):
                        theta[i] = acos(mycos)
                elif  mycos < -1 :       
                    for i in range(count, count + f + 1):
                        theta[i] = pi
                else:
                    for i in range(count, count + f + 1):
                        theta[i] = 0
                count = count + f

    
    Theta = theta[0:count - 1]
    
    return Theta  

def disorientationSYM(np.ndarray[np.int64_t, ndim=2] I, np.ndarray[np.float64_t, ndim=3] Rot, int sym , int type):
    cdef int i, j, ii, jj, p, m, n, count, low, high, l1, l2, symnum, symcomb, f
    cdef float sum
    p = Rot.shape[0]
    m = I.shape[0]
    n = I.shape[1]
    times = p * (p - 1) / 2
    cdef np.ndarray[np.int64_t, ndim = 2] Freq = np.zeros((p, p), dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] theta = np.zeros(times, dtype=np.float64)

    if sym == 4:
        symnum = 24
        T = TMAT.TmatricesCubic()
    elif sym == 6:
        symnum = 12
        T = TMAT.TmatricesHexa()
   
    symcomb = symnum ** 2
   
    cdef np.ndarray[np.float64_t, ndim = 1] mycos = np.zeros(symcomb , dtype=np.float64)
    
    Tt = np.transpose(T, (0, 2, 1))
    mycos = np.zeros(symcomb)
    TT = np.zeros((symcomb, 3, 3))
  
    # COMPUTING ALL COMBINATIONS OF T TRANSPOSE T
    count = 0
    for i in xrange(symnum):
        for j in xrange(symnum):
            TT[count, :, :] = np.dot(Tt[i, :, :], T[j, :, :])
            count = count + 1
    
    
    
    if type == 1: 
        count = 0
        for i in range(m):
            for j in range(n - 1):
                l1 = I[i, j] - 1
                l2 = I[i, j + 1] - 1
                if(l1 != l2):
                    if l1 < l2:
                        low = l1
                        high = l2
                    else:
                        low = l2
                        high = l1
                    Freq[low, high] = Freq[low, high] + 1
                    count = count + 1
        
        
        for j in range(n):
            for i in range(m - 1):
                l1 = I[i, j] - 1
                l2 = I[i + 1, j] - 1
                if(l1 != l2):
                    if l1 < l2:
                        low = l1
                        high = l2
                    else:
                        low = l2
                        high = l1
                    Freq[low, high] = Freq[low, high] + 1
                    count = count + 1
      
    else:
        for ii in range(p - 1):
            for jj in range(ii + 1, p):
                Freq[ii, jj] = 1
    

    count = 0
    Rott = np.transpose(Rot, (0, 2, 1))
    for ii in range(p - 1):
        lasttwo = np.einsum('kil,lj->kij', TT, Rot[ii, :, :])
        for jj in range(ii + 1, p):
            f = Freq[ii, jj] 
            if(f > 0):
                delR = np.einsum('il,klj->kij', Rott[jj, :, :], lasttwo)            
                mycos = (np.einsum('kii', delR) - 1) / 2
                A = np.max(mycos)
                theta[count:count + f ] = acos(A)
                count = count + f
     
    Theta = theta[0:count - 1]
       
    return Theta 


    
    
    
    
    
    
    
    
    
    
    
    
     
    
    


      
