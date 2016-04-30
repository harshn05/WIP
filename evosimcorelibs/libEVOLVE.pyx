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
     

# cdef cutoff(np.ndarray[np.int64_t, ndim=1] X, np.ndarray[np.int64_t, ndim=1] Y, p):
#     
#     cdef np.ndarray[np.int64_t, ndim = 1] XX = np.zeros(p, dtype=np.int64)
#     cdef np.ndarray[np.int64_t, ndim = 1] YY = np.zeros(p, dtype=np.int64)
#     
#     if X.shape[0] == p:
#         return X, Y
#     else:
#         for i in range(p):
#             XX[i] = X[i]
#             YY[i] = Y[i]
#         return XX, YY
            

    
def Evolve_2D_Isotropic_SiteSaturated_without_gr2D(obj):
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef long long asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    
    # Declaring other variables
    cdef double tic, toc
    cdef long long i, j, deli, delj, xo, yo, PN, PE, PW, PS , k, r, Io, Iter
    cdef float red, green, blue 
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()
         
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)        
    r = 0
    Iter = 1
    while True:
        for k in range(0, p):
            I[X[k], Y[k]] = k + 1
            Col[X[k], Y[k], 2] = col[k, 0]
            Col[X[k], Y[k], 1] = col[k, 1]
            Col[X[k], Y[k], 0] = col[k, 2]
        
        showriteframe(sa, sf, fd, r, I, Col, p, Iter, framepause)
        r = 1
        
        while cv2.countNonZero(a) > 0:
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]
    
                    for deli in range(r + 1):
                        for delj in range(r + 1):
                            if deli * deli + delj * delj <= r * r:
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if PN == Io or PW == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue  
                                            a[k] = 1
    
                                i, j = xo - deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue 
                                            a[k] = 1
    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if PE == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue 
                                            a[k] = 1
    
                                i, j = xo + deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if PN == Io or PE == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue 
                                            a[k] = 1
                                            
            showriteframe(sa, sf, fd, r, I, Col, p, Iter, framepause)
            r = r + 1
        
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1
      
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.Col = Col
    return obj
    
def Evolve_2D_Isotropic_SiteSaturated_with_gr2D(obj):
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef long long asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    gr2D = obj.gr2D
       
    # Declaring other variables
    cdef double tic, toc
    cdef long i, j, deli, delj, xo, yo, PN, PE, PW, PS , k, Io, countim, win, xmin, xmax, ymin, ymax, rmax, Iter
    cdef float rad, dr, tmp, gr2DValmax, red, green, blue    
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] gr2DVal = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] r = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()    
       
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)        
    
  
    
    Iter = 1 
    while True:
        for k in range(0, p):
            I[X[k], Y[k]] = k + 1
            Col[X[k], Y[k], 2] = col[k, 0]
            Col[X[k], Y[k], 1] = col[k, 1]
            Col[X[k], Y[k], 0] = col[k, 2]
            gr2DVal[k] = gr2D(X[k], Y[k], k + 1)
          
        gr2DValmax = max(gr2DVal)
        countim = 0
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = 1
        rmax = 0
          
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        r = np.zeros(p, dtype=np.float64)
          
        while cv2.countNonZero(a) > 0: 
       
            gr2DValmax = 0
            for i in range(0, p):
                if a[i] == 1:
                    tmp = gr2DVal[i] 
                    if tmp > gr2DValmax:         
                        gr2DValmax = tmp
              
                 
            for k in range(0, p):
                if a[k] == 1:                    
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]                                                                                        
                    dr = gr2DVal[k] / gr2DValmax
                    r[k] = r[k] + dr
                    rad = r[k]
                    a[k] = 0
                    rmax = math.ceil(rad)
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]  
                       
                    # OUTWARD                                   
                    for deli in range(rmax + 1):                   
                        for delj in range(rmax + 1):
                                                                   
                            if deli ** 2 + delj ** 2 <= (rad + 0.5) ** 2:
                                                                                                                      
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:                                    
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if PN == Io or PE == Io or PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
                                       
                                               
                                                                                                                                    
                                         
           
                                i, j = xo - deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:                                    
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if PN == Io or PE == Io or PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
                                       
                                                                                          
                                           
                                         
           
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:                                    
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if PN == Io or PE == Io or PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1                                        
                                                                  
                                                   
                                         
           
                                i, j = xo + deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:                                    
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if PN == Io or PE == Io or PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
                       
                                
                    if a[k] == 0:
                        win = rmax
                        xmin = max(0, xo - win)
                        xmax = min(m - 1, xo + win)
                        ymin = max(0, yo - win)
                        ymax = min(n - 1, yo + win)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                           
                                      
                                                         
            showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
            countim = countim + 1
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1
           
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.gr2DVal = gr2DVal
    obj.Col = Col
    return obj
   
   
def Evolve_2D_Isotropic_Continuous_without_gr2D(obj):
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float fstop = obj.fstop
    cdef long long MN = m * n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause 
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    NdotIsoXY = obj.NdotIsoXY
    GtIsoXY = obj.GtIsoXY    
    fd = obj.fd
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long p, deli, delj, PN, PE, PW, PS, Io, count, dp, nnz, rad, countim, w, xt, yt, i, j, k, xo, yo, Iter
    cdef float dt, f, t, dr, tmp, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.zeros(MN, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] r = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time()
   
    
    t = 0
    p = 0
    dp = 0
    countim = 0
    nnz = 0
      
    Iter = 1
    while nnz < MN:
        f = nnz / MN
        dt = 1.0 / GtIsoXY(t)
        dp = int((1 - f) * NdotIsoXY(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                count = 0
                while count < dp:
                    xt = np.random.randint(0, m - 1)
                    yt = np.random.randint(0, n - 1)
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        r[p + count] = 1
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                p = p + dp
        else:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                w = 0
                count = 0
                while count < dp:
                    xt = xtrial[w]
                    yt = ytrial[w]
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        r[p + count] = 1
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                    if w < dp - 1:
                        w = w + 1                    
                    else:
                        xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                        w = 0
                p = p + dp 
               
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
   
        for k in range(0, p):
            if a[k] == 1:
                a[k] = 0
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]
                rad = r[k]
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]
   
                for deli in range(rad + 1):
                    for delj in range(rad + 1):
                        if (deli) ** 2 + (delj) ** 2 <= (rad + 0.5) ** 2:
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if PN == Io or PW == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
   
                            i, j = xo - deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
   
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if PE == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
   
                            i, j = xo + deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if PN == Io or PE == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
           
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        for i in range(p):
            r[i] = r[i] + 1
        nnz = cv2.countNonZero(I)
      
    
    
    obj.p = p
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.Col = Col
    return obj
   
   
def Evolve_2D_Isotropic_Continuous_with_gr2D(obj):
       
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float fstop = obj.fstop
    cdef long long MN = m * n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause 
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    NdotIsoXY = obj.NdotIsoXY
    GtIsoXY = obj.GtIsoXY    
    fd = obj.fd
    gr2D = obj.gr2D
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long p, deli, delj, PN, PE, PW, PS, Io, count, xt, yt, dp, nnz, countim, rmax, win, xmin, xmax, ymin, ymax, i, j, k, xo, yo, w, Iter
    cdef float gr2DValmax, dr, rad, tmp, dt, f, t, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.zeros(MN, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] r = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] gr2DVal = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] rold = np.zeros(MN, dtype=np.float64)    
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time()
    t = 0
    p = 0
    dp = 0
    countim = 0
    nnz = cv2.countNonZero(I)
      
    Iter = 1
    while nnz < MN:
        f = nnz / MN
        dt = 1.0 / GtIsoXY(t)
        dp = int((1 - f) * NdotIsoXY(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                count = 0
                while count < dp:
                    xt = np.random.randint(0, m - 1)
                    yt = np.random.randint(0, n - 1)
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        gr2DVal[p + count] = gr2D(xt, yt, p + count + 1)
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                p = p + dp
        else:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                w = 0
                count = 0
                while count < dp:
                    xt = xtrial[w]
                    yt = ytrial[w]
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        gr2DVal[p + count] = gr2D(xt, yt, p + count + 1)
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                    if w < dp - 1:
                        w = w + 1                    
                    else:
                        xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                        w = 0
                p = p + dp
               
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
            
        gr2DValmax = 0
        for i in range(0, p):
            if a[i] == 1:
                tmp = gr2DVal[i] 
                if tmp > gr2DValmax:         
                    gr2DValmax = tmp
    
        for k in range(0, p):
            if a[k] == 1:             
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]                                                                                        
                dr = gr2DVal[k] / gr2DValmax
                r[k] = r[k] + dr
                rad = r[k]
                a[k] = 0
                rmax = math.ceil(rad)
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]                                   
                for deli in range(0, rmax + 1):
                    for delj in range (0, rmax + 1):
                        if deli ** 2 + delj ** 2 <= (rad + 0.5) ** 2:
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                        
                                    if PN == Io or PE == Io or PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                
                            i, j = xo - deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                        
                                    if PN == Io or PE == Io or PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                        
                                    if PN == Io or PE == Io or PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                
                            i, j = xo + deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                                                                    
                                    if PN == Io or PE == Io or PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                if a[k] == 0:
                    win = rmax
                    xmin = max(0, xo - win)
                    xmax = min(m - 1, xo + win)
                    ymin = max(0, yo - win)
                    ymax = min(n - 1, yo + win)
                    a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)                 
           
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        nnz = cv2.countNonZero(I)
           
      
     
    obj.p = p    
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.gr2DVal = gr2DVal
    obj.Col = Col   
    
    return obj   
   
   
def Evolve_2D_Anisotropic_SiteSaturated_NeighbourHoodBased_N4_without_gr2D(obj):
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long i, j, deli, delj, xo, yo, PN, PE, PW, PS , k, r, Io, Iter
    cdef float red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()
            
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)        
    r = 0
      
    Iter = 1
    while True:
              
        for k in range(0, p):
            I[X[k], Y[k]] = k + 1
            Col[X[k], Y[k], 2] = col[k, 0]
            Col[X[k], Y[k], 1] = col[k, 1]
            Col[X[k], Y[k], 0] = col[k, 2]
               
        showriteframe(sa, sf, fd, r, I, Col, p, Iter, framepause)
        r = 1
           
        while cv2.countNonZero(a) > 0:
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]                
       
                    for deli in range(r + 1):
                        delj = r - deli
                        i, j = xo + deli, yo + delj
                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                            if I[i, j] == 0:
                                if i > 0:
                                    PN = I[i - 1, j]
                                else:
                                    PN = 0
                                if j > 0:
                                    PW = I[i, j - 1]
                                else:
                                    PW = 0
                                if PN == Io or PW == Io:
                                    I[i, j] = Io
                                    Col[i, j, 2] = red
                                    Col[i, j, 1] = green 
                                    Col[i, j, 0] = blue
                                    a[k] = 1
       
                        i, j = xo - deli, yo + delj
                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                            if I[i, j] == 0:
                                if i < m - 1:
                                    PS = I[i + 1, j]
                                else:
                                    PS = 0
                                if j > 0:
                                    PW = I[i, j - 1]
                                else:
                                    PW = 0
                                if PW == Io or PS == Io:
                                    I[i, j] = Io
                                    Col[i, j, 2] = red
                                    Col[i, j, 1] = green 
                                    Col[i, j, 0] = blue
                                    a[k] = 1
       
                        i, j = xo - deli, yo - delj
                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                            if I[i, j] == 0:
                                if i < m - 1:
                                    PS = I[i + 1, j]
                                else:
                                    PS = 0
                                if j < n - 1:
                                    PE = I[i, j + 1]
                                else:
                                    PE = 0
                                if PE == Io or PS == Io:
                                    I[i, j] = Io
                                    Col[i, j, 2] = red
                                    Col[i, j, 1] = green 
                                    Col[i, j, 0] = blue
                                    a[k] = 1
       
                        i, j = xo + deli, yo - delj
                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                            if I[i, j] == 0:
                                if i > 0:
                                    PN = I[i - 1, j]
                                else:
                                    PN = 0
                                if j < n - 1:
                                    PE = I[i, j + 1]
                                else:
                                    PE = 0
                                if PN == Io or PE == Io:
                                    I[i, j] = Io
                                    Col[i, j, 2] = red
                                    Col[i, j, 1] = green 
                                    Col[i, j, 0] = blue
                                    a[k] = 1
            showriteframe(sa, sf, fd, r, I, Col, p, Iter, framepause)
            r = r + 1
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1
      
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.Col = Col
    return obj
   
   
def Evolve_2D_Anisotropic_SiteSaturated_NeighbourHoodBased_N8_without_gr2D(obj):
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long i, j, deli, delj, xo, yo, PN, PE, PW, PS , k, r, Io, Iter
    cdef float red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()
            
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)        
    r = 0
      
    Iter = 1
    while True:
      
        for k in range(0, p):
            I[X[k], Y[k]] = k + 1
            Col[X[k], Y[k], 2] = col[k, 0]
            Col[X[k], Y[k], 1] = col[k, 1]
            Col[X[k], Y[k], 0] = col[k, 2]
               
        showriteframe(sa, sf, fd, r, I, Col, p, Iter, framepause)
        r = 1
          
      
        while cv2.countNonZero(a) > 0:
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]
       
                    for deli in range(r + 1):
                        delj = r
                        i, j = xo + deli, yo + delj
                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                            if I[i, j] == 0:
                                if i > 0:
                                    PN = I[i - 1, j]
                                else:
                                    PN = 0
                                if j > 0:
                                    PW = I[i, j - 1]
                                else:
                                    PW = 0
                                if PN == Io or PW == Io:
                                    I[i, j] = Io
                                    Col[i, j, 2] = red
                                    Col[i, j, 1] = green 
                                    Col[i, j, 0] = blue
                                    a[k] = 1
       
                        i, j = xo - deli, yo + delj
                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                            if I[i, j] == 0:
                                if i < m - 1:
                                    PS = I[i + 1, j]
                                else:
                                    PS = 0
                                if j > 0:
                                    PW = I[i, j - 1]
                                else:
                                    PW = 0
                                if PW == Io or PS == Io:
                                    I[i, j] = Io
                                    Col[i, j, 2] = red
                                    Col[i, j, 1] = green 
                                    Col[i, j, 0] = blue
                                    a[k] = 1
       
                        i, j = xo - deli, yo - delj
                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                            if I[i, j] == 0:
                                if i < m - 1:
                                    PS = I[i + 1, j]
                                else:
                                    PS = 0
                                if j < n - 1:
                                    PE = I[i, j + 1]
                                else:
                                    PE = 0
                                if PE == Io or PS == Io:
                                    I[i, j] = Io
                                    Col[i, j, 2] = red
                                    Col[i, j, 1] = green 
                                    Col[i, j, 0] = blue
                                    a[k] = 1
       
                        i, j = xo + deli, yo - delj
                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                            if I[i, j] == 0:
                                if i > 0:
                                    PN = I[i - 1, j]
                                else:
                                    PN = 0
                                if j < n - 1:
                                    PE = I[i, j + 1]
                                else:
                                    PE = 0
                                if PN == Io or PE == Io:
                                    I[i, j] = Io
                                    Col[i, j, 2] = red
                                    Col[i, j, 1] = green 
                                    Col[i, j, 0] = blue
                                    a[k] = 1
                                       
                           
                        # INTERCHANGE deli and delj
                        i, j = xo + delj, yo + deli
                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                            if I[i, j] == 0:
                                if i > 0:
                                    PN = I[i - 1, j]
                                else:
                                    PN = 0
                                if j > 0:
                                    PW = I[i, j - 1]
                                else:
                                    PW = 0
                                if PN == Io or PW == Io:
                                    I[i, j] = Io
                                    Col[i, j, 2] = red
                                    Col[i, j, 1] = green 
                                    Col[i, j, 0] = blue
                                    a[k] = 1
       
                        i, j = xo - delj, yo + deli
                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                            if I[i, j] == 0:
                                if i < m - 1:
                                    PS = I[i + 1, j]
                                else:
                                    PS = 0
                                if j > 0:
                                    PW = I[i, j - 1]
                                else:
                                    PW = 0
                                if PW == Io or PS == Io:
                                    I[i, j] = Io
                                    Col[i, j, 2] = red
                                    Col[i, j, 1] = green 
                                    Col[i, j, 0] = blue
                                    a[k] = 1
       
                        i, j = xo - delj, yo - deli
                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                            if I[i, j] == 0:
                                if i < m - 1:
                                    PS = I[i + 1, j]
                                else:
                                    PS = 0
                                if j < n - 1:
                                    PE = I[i, j + 1]
                                else:
                                    PE = 0
                                if PE == Io or PS == Io:
                                    I[i, j] = Io
                                    Col[i, j, 2] = red
                                    Col[i, j, 1] = green 
                                    Col[i, j, 0] = blue
                                    a[k] = 1
       
                        i, j = xo + delj, yo - deli
                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                            if I[i, j] == 0:
                                if i > 0:
                                    PN = I[i - 1, j]
                                else:
                                    PN = 0
                                if j < n - 1:
                                    PE = I[i, j + 1]
                                else:
                                    PE = 0
                                if PN == Io or PE == Io:
                                    I[i, j] = Io
                                    Col[i, j, 2] = red
                                    Col[i, j, 1] = green 
                                    Col[i, j, 0] = blue
                                    a[k] = 1
            showriteframe(sa, sf, fd, r, I, Col, p, Iter, framepause)
            r = r + 1
           
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1
     
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.Col = Col
    return obj
   
   
def Evolve_2D_Anisotropic_SiteSaturated_NeighbourHoodBased_N4_with_gr2D(obj):
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    gr2D = obj.gr2D
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long i, j, deli, delj, xo, yo, PN, PE, PW, PS , k, Io, countim, win, xmin, xmax, ymin, ymax, rmax, Iter
    cdef float rad, dr, tmp, gr2DValmax, red, green, blue    
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] gr2DVal = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] r = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()    
       
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)        
      
    Iter = 1
    while True:
        for k in range(0, p):
            I[X[k], Y[k]] = k + 1
            gr2DVal[k] = gr2D(X[k], Y[k], k + 1)
            Col[X[k], Y[k], 2] = col[k, 0]
            Col[X[k], Y[k], 1] = col[k, 1]
            Col[X[k], Y[k], 0] = col[k, 2]
               
        gr2DValmax = max(gr2DVal)
        countim = 0
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = 1
        rmax = 0
        Io = 0; xo = 0; yo = 0;
           
        while cv2.countNonZero(a) > 0: 
       
            gr2DValmax = 0
            for i in range(0, p):
                if a[i] == 1:
                    tmp = gr2DVal[i] 
                    if tmp > gr2DValmax:         
                        gr2DValmax = tmp
                 
            for k in range(0, p):
                if a[k] == 1:                    
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]                                                                                        
                    dr = gr2DVal[k] / gr2DValmax
                    r[k] = r[k] + dr
                    rad = r[k]
                    a[k] = 0
                    rmax = math.ceil(rad)
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]  
                       
                    # OUTWARD                                   
                    for deli in range(rmax + 1):                   
                        for delj in range(rmax + 1):
                                                                   
                            if deli + delj <= rad :
                                                                                                                      
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:                                    
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if PN == Io or PE == Io or PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
                                       
                                               
                                                                                                                                    
                                         
           
                                i, j = xo - deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:                                    
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if PN == Io or PE == Io or PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
                                       
                                                                                          
                                           
                                         
           
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:                                    
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if PN == Io or PE == Io or PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1                                        
                                                                  
                                                   
                                         
           
                                i, j = xo + deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:                                    
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if PN == Io or PE == Io or PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
                    if a[k] == 0:
                        win = rmax
                        xmin = max(0, xo - win)
                        xmax = min(m - 1, xo + win)
                        ymin = max(0, yo - win)
                        ymax = min(n - 1, yo + win)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                           
                                      
                                                         
            showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
            countim = countim + 1
         
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                r = np.zeros(p, dtype=np.float64)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1
           
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.gr2DVal = gr2DVal
    obj.Col = Col
    return obj
   
   
def Evolve_2D_Anisotropic_SiteSaturated_NeighbourHoodBased_N8_with_gr2D(obj):
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    gr2D = obj.gr2D
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long i, j, deli, delj, xo, yo, PN, PE, PW, PS , k, Io, countim, win, xmin, xmax, ymin, ymax, rmax, Iter
    cdef float rad, dr, tmp, gr2DValmax, red, green, blue    
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] gr2DVal = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] r = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()    
       
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)        
    
      
    Iter = 1
    while True:
        for k in range(0, p):
            I[X[k], Y[k]] = k + 1
            gr2DVal[k] = gr2D(X[k], Y[k], k + 1)
            Col[X[k], Y[k], 2] = col[k, 0]
            Col[X[k], Y[k], 1] = col[k, 1]
            Col[X[k], Y[k], 0] = col[k, 2]
               
        gr2DValmax = max(gr2DVal)
        countim = 0
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = 1
        rmax = 0
        Io = 0; xo = 0; yo = 0;
           
        while cv2.countNonZero(a) > 0: 
       
            gr2DValmax = 0
            for i in range(0, p):
                if a[i] == 1:
                    tmp = gr2DVal[i] 
                    if tmp > gr2DValmax:         
                        gr2DValmax = tmp
                 
            for k in range(0, p):
                if a[k] == 1:                    
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]                                                                                        
                    dr = gr2DVal[k] / gr2DValmax
                    r[k] = r[k] + dr
                    rad = r[k]
                    a[k] = 0
                    rmax = math.ceil(rad)
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]  
                       
                    # OUTWARD                                   
                    for deli in range(rmax + 1):                   
                        for delj in range(rmax + 1):
                                                                   
                            if deli <= rad and delj <= rad :
                                                                                                                      
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:                                    
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if PN == Io or PE == Io or PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
                                       
                                               
                                                                                                                                    
                                         
           
                                i, j = xo - deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:                                    
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if PN == Io or PE == Io or PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
                                       
                                                                                          
                                           
                                         
           
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:                                    
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if PN == Io or PE == Io or PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1                                        
                                                                  
                                                   
                                         
           
                                i, j = xo + deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:                                    
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if PN == Io or PE == Io or PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
                    if a[k] == 0:
                        win = rmax
                        xmin = max(0, xo - win)
                        xmax = min(m - 1, xo + win)
                        ymin = max(0, yo - win)
                        ymax = min(n - 1, yo + win)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                           
                                      
                                                         
            showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
            countim = countim + 1
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                r = np.zeros(p, dtype=np.float64)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1         
  
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.gr2DVal = gr2DVal
    obj.Col = Col
    return obj
   
   
def Evolve_2D_Anisotropic_Continuous_NeighbourHoodBased_N4_without_gr2D(obj):
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float fstop = obj.fstop
    cdef long long MN = m * n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause 
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Ndot = obj.Ndot
    Gt = obj.Gt    
    fd = obj.fd
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long p, deli, delj, PN, PE, PW, PS, Io, count, dp, nnz, rad, countim, w, xt, yt, i, j, k, xo, yo, Iter
    cdef float dt, f, t, dr, tmp, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.zeros(MN, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] r = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time()
    
    t = 0
    p = 0
    dp = 0
    countim = 0
    nnz = 0
    Iter = 1
    while nnz < MN:
        f = nnz / MN
        dt = 1.0 / Gt(t)
        dp = int((1 - f) * Ndot(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                count = 0
                while count < dp:
                    xt = np.random.randint(0, m - 1)
                    yt = np.random.randint(0, n - 1)
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        r[p + count] = 1
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                p = p + dp
        else:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                w = 0
                count = 0
                while count < dp:
                    xt = xtrial[w]
                    yt = ytrial[w]
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        r[p + count] = 1
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                    if w < dp - 1:
                        w = w + 1                    
                    else:
                        xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                        w = 0
                p = p + dp 
          
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
   
        for k in range(0, p):
            if a[k] == 1:
                a[k] = 0
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]
                rad = r[k]
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]
   
                for deli in range(rad + 1):
                    for delj in range(rad + 1):
                        if deli + delj <= rad:
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if PN == Io or PW == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
   
                            i, j = xo - deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
   
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if PE == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
   
                            i, j = xo + deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if PN == Io or PE == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
           
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        for i in range(p):
            r[i] = r[i] + 1
        nnz = cv2.countNonZero(I)
      
    obj.p = p
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.Col = Col
    return obj
       
   
def Evolve_2D_Anisotropic_Continuous_NeighbourHoodBased_N8_without_gr2D(obj):
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float fstop = obj.fstop
    cdef long long MN = m * n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause 
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Ndot = obj.Ndot
    Gt = obj.Gt    
    fd = obj.fd
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long p, deli, delj, PN, PE, PW, PS, Io, count, dp, nnz, rad, countim, w, xt, yt, i, j, k, xo, yo, Iter
    cdef float dt, f, t, dr, tmp, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.zeros(MN, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] r = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time() 
    
    
    t = 0
    p = 0
    dp = 0
    countim = 0
    nnz = 0
      
    Iter = 1
    while nnz < MN:
        f = nnz / MN
        dt = 1.0 / Gt(t)
        dp = int((1 - f) * Ndot(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                count = 0
                while count < dp:
                    xt = np.random.randint(0, m - 1)
                    yt = np.random.randint(0, n - 1)
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        r[p + count] = 1
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                p = p + dp
        else:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                w = 0
                count = 0
                while count < dp:
                    xt = xtrial[w]
                    yt = ytrial[w]
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        r[p + count] = 1
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                    if w < dp - 1:
                        w = w + 1                    
                    else:
                        xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                        w = 0
                p = p + dp  
          
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
   
        for k in range(0, p):
            if a[k] == 1:
                a[k] = 0
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]
                rad = r[k]
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]
   
                for deli in range(rad + 1):
                    for delj in range(rad + 1):
                        if deli <= rad and delj <= rad:
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if PN == Io or PW == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
   
                            i, j = xo - deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
   
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if PE == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
   
                            i, j = xo + deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if PN == Io or PE == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
           
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        for i in range(p):
            r[i] = r[i] + 1
        nnz = cv2.countNonZero(I)
      
    obj.p = p
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.Col = Col
    return obj
   
   
def Evolve_2D_Anisotropic_Continuous_NeighbourHoodBased_N4_with_gr2D(obj):
       
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float fstop = obj.fstop
    cdef long long MN = m * n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause 
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Ndot = obj.Ndot
    Gt = obj.Gt    
    fd = obj.fd
    gr2D = obj.gr2D
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long p, deli, delj, PN, PE, PW, PS, Io, count, xt, yt, dp, nnz, countim, rmax, win, xmin, xmax, ymin, ymax, i, j, k, xo, yo, w, Iter
    cdef float gr2DValmax, dr, rad, tmp, dt, f, t, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.zeros(MN, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] r = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] gr2DVal = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] rold = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time() 
   
     
           
    t = 0
    p = 0
    dp = 0
    countim = 0
    nnz = cv2.countNonZero(I)
    Iter = 1
    while nnz < MN:
        f = nnz / MN
        dt = 1.0 / Gt(t)
        dp = int((1 - f) * Ndot(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                count = 0
                while count < dp:
                    xt = np.random.randint(0, m - 1)
                    yt = np.random.randint(0, n - 1)
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        gr2DVal[p + count] = gr2D(xt, yt, p + count + 1)
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                p = p + dp
        else:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                w = 0
                count = 0
                while count < dp:
                    xt = xtrial[w]
                    yt = ytrial[w]
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        gr2DVal[p + count] = gr2D(xt, yt, p + count + 1)
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                    if w < dp - 1:
                        w = w + 1                    
                    else:
                        xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                        w = 0
                p = p + dp
           
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
            
        gr2DValmax = 0
        for i in range(0, p):
            if a[i] == 1:
                tmp = gr2DVal[i] 
                if tmp > gr2DValmax:         
                    gr2DValmax = tmp
    
        for k in range(0, p):
            if a[k] == 1:             
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]                                                                                        
                dr = gr2DVal[k] / gr2DValmax
                r[k] = r[k] + dr
                rad = r[k]
                a[k] = 0
                rmax = math.ceil(rad)
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]                                   
                for deli in range(0, rmax + 1):
                    for delj in range (0, rmax + 1):
                        if deli + delj <= rad:
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                        
                                    if PN == Io or PE == Io or PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                
                            i, j = xo - deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                        
                                    if PN == Io or PE == Io or PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                        
                                    if PN == Io or PE == Io or PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                
                            i, j = xo + deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                                                                    
                                    if PN == Io or PE == Io or PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                if a[k] == 0:
                    win = rmax
                    xmin = max(0, xo - win)
                    xmax = min(m - 1, xo + win)
                    ymin = max(0, yo - win)
                    ymax = min(n - 1, yo + win)
                    a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)                 
           
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        nnz = cv2.countNonZero(I)
           
      
    obj.p = p    
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.gr2DVal = gr2DVal
    obj.Col = Col   
    
    return obj   
   
   
def Evolve_2D_Anisotropic_Continuous_NeighbourHoodBased_N8_with_gr2D(obj):
       
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float fstop = obj.fstop
    cdef long long MN = m * n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause 
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Ndot = obj.Ndot
    Gt = obj.Gt    
    fd = obj.fd
    gr2D = obj.gr2D
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long p, deli, delj, PN, PE, PW, PS, Io, count, xt, yt, dp, nnz, countim, rmax, win, xmin, xmax, ymin, ymax, i, j, k, xo, yo, w, Iter
    cdef float gr2DValmax, dr, rad, tmp, dt, f, t, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.zeros(MN, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] r = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] gr2DVal = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] rold = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time()
       
       
           
    t = 0
    p = 0
    dp = 0
    countim = 0
    nnz = cv2.countNonZero(I)
    Iter = 1
    while nnz < MN:
        f = nnz / MN
        dt = 1.0 / Gt(t)
        dp = int((1 - f) * Ndot(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                count = 0
                while count < dp:
                    xt = np.random.randint(0, m - 1)
                    yt = np.random.randint(0, n - 1)
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        gr2DVal[p + count] = gr2D(xt, yt, p + count + 1)
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                p = p + dp
        else:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                w = 0
                count = 0
                while count < dp:
                    xt = xtrial[w]
                    yt = ytrial[w]
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        gr2DVal[p + count] = gr2D(xt, yt, p + count + 1)
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                    if w < dp - 1:
                        w = w + 1                    
                    else:
                        xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                        w = 0
                p = p + dp
               
           
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
            
        gr2DValmax = 0
        for i in range(0, p):
            if a[i] == 1:
                tmp = gr2DVal[i] 
                if tmp > gr2DValmax:         
                    gr2DValmax = tmp
    
        for k in range(0, p):
            if a[k] == 1:             
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]                                                                                        
                dr = gr2DVal[k] / gr2DValmax
                r[k] = r[k] + dr
                rad = r[k]
                a[k] = 0
                rmax = math.ceil(rad)
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]                                   
                for deli in range(0, rmax + 1):
                    for delj in range (0, rmax + 1):
                        if deli <= rad and delj <= rad:
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                        
                                    if PN == Io or PE == Io or PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                
                            i, j = xo - deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                        
                                    if PN == Io or PE == Io or PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                        
                                    if PN == Io or PE == Io or PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                
                            i, j = xo + deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                                                                    
                                    if PN == Io or PE == Io or PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                if a[k] == 0:
                    win = rmax
                    xmin = max(0, xo - win)
                    xmax = min(m - 1, xo + win)
                    ymin = max(0, yo - win)
                    ymax = min(n - 1, yo + win)
                    a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)                 
           
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        nnz = cv2.countNonZero(I)
           
       
    obj.p = p    
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.gr2DVal = gr2DVal
    obj.Col = Col   
    
    return obj
   
def Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_without_aspect_without_theta_without_adot(obj):
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float R = obj.R
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    fd = obj.fd
    pdelNxy = obj.pdelNxy
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long i, j, deli, delj, xo, yo, PN, PE, PW, PS , k, Io, MAJceil, MINceil, Iter
    cdef float MAJ, MIN, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()
            
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)        
    MAJceil = 0
      
    Iter = 1
    while True:
        for k in range(0, p):
            I[X[k], Y[k]] = k + 1
            Col[X[k], Y[k], 2] = col[k, 0]
            Col[X[k], Y[k], 1] = col[k, 1]
            Col[X[k], Y[k], 0] = col[k, 2]
               
        showriteframe(sa, sf, fd, MAJceil, I, Col, p, Iter, framepause)
           
        MAJ = 1.0
        MIN = MAJ / R
        
           
        while cv2.countNonZero(a) > 0:
            MAJceil = math.ceil(MAJ)
            MINceil = math.ceil(MIN)
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]
        
                    for deli in range(MAJceil + 1):
                        for delj in range(MINceil + 1):
                            if (deli / MAJ) ** 2 + (delj / MIN) ** 2 <= 1:
                                   
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if PN == Io or PW == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
        
                                i, j = xo - deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
        
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if PE == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
        
                                i, j = xo + deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if PN == Io or PE == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
                                                
                    if a[k] == 0:
                        xmin = max(0, xo - MAJceil)
                        xmax = min(m - 1, xo + MAJceil)
                        ymin = max(0, yo - MINceil)
                        ymax = min(n - 1, yo + MINceil)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                
            showriteframe(sa, sf, fd, MAJceil, I, Col, p, Iter, framepause)
            MAJ = MAJ + 1.0
            MIN = MAJ / R
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1      
       
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.Col = Col
    return obj
   
def Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_with_aspect_without_theta_without_adot(obj):
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Rfunc = obj.Rfunc
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long deli, delj, PN, PE, PW, PS, Io, win, xmin, xmax, ymin, ymax, dum, count, l, l1 , l2, imax, imin, jmax, jmin, io, jo, go, found, i, j, k, xo, yo, MAJceil, MINceil, Iter
    cdef float MAJ, MIN, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] II = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Rval = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()
   
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)
      
    Iter = 1
      
    while True:
       
        for k in range(0, p):
            I[X[k], Y[k]] = k + 1
            Rval[k] = Rfunc(X[k], Y[k], k + 1, 1, 0)
            Col[X[k], Y[k], 2] = col[k, 0]
            Col[X[k], Y[k], 1] = col[k, 1]
            Col[X[k], Y[k], 0] = col[k, 2]
                
        showriteframe(sa, sf, fd, 0, I, Col, p, Iter, framepause)
        MAJ = 1.0
           
        while cv2.countNonZero(a) > 0:
            MAJceil = math.ceil(MAJ)
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]
                        
                    MIN = MAJ / Rval[k]
                    MINceil = math.ceil(MIN)
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]              
                        
                        
                    imax, imin, jmax, jmin = xo, xo, yo, yo
                                                 
                    count = 0
                    for deli in range(0, MAJceil + 1):
                        for delj in range(0, MINceil + 1):                
                            if (deli / MAJ) ** 2 + (delj / MIN) ** 2 <= 1:
                                for l1 in range(-1, 2, 2):
                                    for l2 in range(-1, 2, 2):
                                        i, j = xo + l1 * deli, yo + l2 * delj
                                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                            if I[i, j] == 0:
                                                II[i, j] = Io
                                                count = count + 1                                            
                                                if i > imax:
                                                    imax = i
                                                if i < imin:
                                                    imin = i
                                                if j > jmax:
                                                    jmax = j
                                                if j < jmin:
                                                    jmin = j
                                                    
                    if count > 0:                                              
                        i = imin
                        go = 1
                        found = 0
                        while go is True:
                            for j in range(jmin, jmax + 1, 1):
                                if II[i, j] == Io:
                                    if i > 0:
                                        if  I[i - 1, j] == Io:
                                            io, jo = i, j
                                            go = 0
                                            found = 1
                                            break
                                    if j > 0:
                                        if I[i, j - 1] == Io:
                                            io, jo = i, j
                                            go = 0
                                            found = 1
                                            break
                                                                     
                                    if j < n - 1:
                                        if I[i, j + 1] == Io:
                                            io, jo = i, j
                                            go = 0
                                            found = 1
                                            break
                                          
                                        
                                    if i < m - 1:
                                        if  I[i + 1, j] == Io:
                                            io, jo = i, j
                                            go = 0
                                            found = 1
                                            break
                            i = i + 1
                            if i == imax + 1:
                                go = 0
                                    
                        if found == 1:
                                
                            # OUTWARD
                            for i in range(io, imax + 1):
                                for j in range(jo, jmax + 1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            
                                for j in range(jo - 1, jmin - 1, -1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                
                                
                                
                            for i in range(io - 1, imin - 1, -1):
                                for j in range(jo, jmax + 1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                    
                                for j in range(jo - 1, jmin - 1, -1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                
                                
                                
                            # INWARD
                            for i in range(imax, io - 1, -1):
                                for j in range(jo, jmax + 1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                    
                                for j in range(jo - 1, jmin - 1, -1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                
                                
                                
                            for i in range(imin, io, -1):
                                for j in range(jo, jmax + 1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                    
                                for j in range(jo - 1, jmin - 1, -1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                
       
                    if a[k] == 0:
                        xmin = max(0, xo - MAJceil)
                        xmax = min(m - 1, xo + MAJceil)
                        ymin = max(0, yo - MINceil)
                        ymax = min(n - 1, yo + MINceil)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
      
            showriteframe(sa, sf, fd, MAJceil, I, Col, p, Iter, framepause)
            MAJ = MAJ + 1.0
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                II = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.Rvalues = Rval
    obj.Col = Col
    return obj 
       
   
def Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_without_aspect_with_theta_without_adot(obj):
          
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float R = obj.R
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Thetafunc = obj.Thetafunc
       
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long deli, delj, PN, PE, PW, PS, Io, win, xmin, xmax, ymin, ymax, dum, countim, i, j, k, xo, yo, delX, delY, Iter
    cdef float MAJ, MIN, theta, c, s, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] Theta = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()
   
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)
      
      
    Iter = 1
      
    while True:
       
        for k in range(0, p):
            I[X[k], Y[k]] = k + 1
            Theta[k] = Thetafunc(X[k], Y[k], k + 1 , 1, 1)
            Col[X[k], Y[k], 2] = col[k, 0]
            Col[X[k], Y[k], 1] = col[k, 1]
            Col[X[k], Y[k], 0] = col[k, 2]
                
        showriteframe(sa, sf, fd, 0, I, Col, p, Iter, framepause)
        MAJ = 1.0
        MIN = MAJ / R
        
        countim = 0
        while cv2.countNonZero(a) > 0:
            countim = countim + 1
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    theta = Theta[k]
                    c = math.cos(theta)
                    s = -math.sin(theta)
                    Io = I[xo, yo]
                    delX = math.ceil(math.sqrt((MAJ * c) ** 2 + (MIN * s) ** 2))
                    delY = math.ceil(math.sqrt((MAJ * s) ** 2 + (MIN * c) ** 2))
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]
        
                    for deli in range(0, delX + 1):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                        
                    for deli in range(0, -delX - 1, -1):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                                    
                                                
                    if a[k] == 0:
                        xmin = max(0, xo - delX)
                        xmax = min(m - 1, xo + delX)
                        ymin = max(0, yo - delY)
                        ymax = min(n - 1, yo + delY)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                
            showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
            MAJ = MAJ + 1.0
            MIN = MAJ / R
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1   
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.Thetavalues = Theta
    obj.Col = Col
    return obj
   
def Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_without_aspect_without_theta_with_adot(obj):
       
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float R = obj.R
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Adotfunc = obj.Adotfunc
       
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long deli, delj, MN, PN, PE, PW, PS, Io, win, xmin, xmax, ymin, ymax, imin, imax, jmin, jmax, io, jo, go, found, dum, i, j, k, xo, yo, count, l1, l2, countim, MAJceil, MINceil, Iter
    cdef float adotValmax, tmp, MIN, dr, theta, c, s, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] adotVal = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim = 2] II = np.zeros((m, n), dtype=np.int)
    cdef np.ndarray[np.float64_t, ndim = 1] MAJ = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()
   
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)
       
    Iter = 1
    while True:
        for k in range(0, p):
            I[X[k], Y[k]] = k + 1
            adotVal[k] = Adotfunc(X[k], Y[k], k + 1 , 0, 1)
            Col[X[k], Y[k], 2] = col[k, 0]
            Col[X[k], Y[k], 1] = col[k, 1]
            Col[X[k], Y[k], 0] = col[k, 2]
                
          
           
        adotValmax = max(adotVal)
                
        showriteframe(sa, sf, fd, 0, I, Col, p, Iter, framepause)
        
        countim = 1
        count = 0
        while cv2.countNonZero(a) > 0:
                
                
            adotValmax = 0
            for i in range(0, p):
                if a[i] == 1:
                    tmp = adotVal[i] 
                    if tmp > adotValmax:         
                        adotValmax = tmp
                
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]
                        
                    dr = adotVal[k] / adotValmax
                    MAJ[k] = MAJ[k] + dr
                    MIN = MAJ[k] / R
                    MAJceil = math.ceil(MAJ[k])
                    MINceil = math.ceil(MIN)
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]
                        
                    imin, imax, jmin, jmax = xo, xo, yo, yo
        
                    for deli in range(0, MAJceil + 1):
                        for delj in range(0, MINceil + 1):                
                            if (deli / MAJ[k]) ** 2 + (delj / MIN) ** 2 <= 1:
                                for l1 in range(-1, 2, 2):
                                    for l2 in range(-1, 2, 2):
                                        i, j = xo + l1 * deli, yo + l2 * delj
                                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                            if I[i, j] == 0:
                                                II[i, j] = Io
                                                count = count + 1                                            
                                                if i > imax:
                                                    imax = i
                                                if i < imin:
                                                    imin = i
                                                if j > jmax:
                                                    jmax = j
                                                if j < jmin:
                                                    jmin = j
                                                    
                    if count > 0:                                              
                        i = imin
                        go = 1
                        found = 0
                        while go is True:
                            for j in range(jmin, jmax + 1, 1):
                                if II[i, j] == Io:
                                    if i > 0:
                                        if  I[i - 1, j] == Io:
                                            io, jo = i, j
                                            go = 0
                                            found = 1
                                            break
                                    if j > 0:
                                        if I[i, j - 1] == Io:
                                            io, jo = i, j
                                            go = 0
                                            found = 1
                                            break
                                                                     
                                    if j < n - 1:
                                        if I[i, j + 1] == Io:
                                            io, jo = i, j
                                            go = 0
                                            found = 1
                                            break
                                          
                                        
                                    if i < m - 1:
                                        if  I[i + 1, j] == Io:
                                            io, jo = i, j
                                            go = 0
                                            found = 1
                                            break
                            i = i + 1
                            if i == imax + 1:
                                go = 0
                                    
                        if found == 1:
                                
                            # OUTWARD
                            for i in range(io, imax + 1):
                                for j in range(jo, jmax + 1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            
                                for j in range(jo - 1, jmin - 1, -1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                
                                
                                
                            for i in range(io - 1, imin - 1, -1):
                                for j in range(jo, jmax + 1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                    
                                for j in range(jo - 1, jmin - 1, -1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                
                                
                                
                            # INWARD
                            for i in range(imax, io - 1, -1):
                                for j in range(jo, jmax + 1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                    
                                for j in range(jo - 1, jmin - 1, -1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                
                                
                                
                            for i in range(imin, io, -1):
                                for j in range(jo, jmax + 1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    a[k] = 1
                                                    break
                                    
                                for j in range(jo - 1, jmin - 1, -1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                
                    if a[k] == 0:
                        xmin = max(0, xo - MAJceil)
                        xmax = min(m - 1, xo + MAJceil)
                        ymin = max(0, yo - MINceil)
                        ymax = min(n - 1, yo + MINceil)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                          
            showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
            countim = countim + 1
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                II = np.zeros((m, n), dtype=np.int)
                MAJ = np.zeros(p, dtype=np.float64)
                a = np.ones(p, dtype=np.int8)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.adotVal = adotVal
    obj.Col = Col
    return obj
    
   
def Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_with_aspect_with_theta_without_adot(obj):
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef long long seq = obj.seq
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Thetafunc = obj.Thetafunc
    Rfunc = obj.Rfunc
       
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long deli, delj, MN, PN, PE, PW, PS, Io, win, xmin, xmax, ymin, ymax, dum, countim, i, j, k, xo, yo, delX, delY, Iter
    cdef float  MIN, theta, c, s, R, MAJ, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] II = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.float64_t, ndim = 1] adotVal = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] Theta = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Rval = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()
    
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)
      
      
    Iter = 1
      
    while True:
        if seq == 1 or seq == 2 or seq == 5: 
            for k in range(0, p):
                I[X[k], Y[k]] = k + 1
                Col[X[k], Y[k], 2] = col[k, 0]
                Col[X[k], Y[k], 1] = col[k, 1]
                Col[X[k], Y[k], 0] = col[k, 2]
                Theta[k] = Thetafunc(X[k], Y[k], k + 1, 1, 1)
                Rval[k] = Rfunc(X[k], Y[k], k + 1, 1, Theta[k])
        else:
            for k in range(0, p):
                I[X[k], Y[k]] = k + 1
                Col[X[k], Y[k], 2] = col[k, 0]
                Col[X[k], Y[k], 1] = col[k, 1]
                Col[X[k], Y[k], 0] = col[k, 2]
                Rval[k] = Rfunc(X[k], Y[k], k + 1, 1, 0)
                Theta[k] = Thetafunc(X[k], Y[k], k + 1, Rval[k], 1)
                   
                
        showriteframe(sa, sf, fd, 0, I, Col, p, Iter, framepause)
        MAJ = 1.0
        countim = 0
        while cv2.countNonZero(a) > 0:
            countim = countim + 1
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    theta = Theta[k]
                    c = math.cos(theta)
                    s = -math.sin(theta)
                    Io = I[xo, yo]
                    MIN = MAJ / Rval[k]
                    delX = math.ceil(math.sqrt((MAJ * c) ** 2 + (MIN * s) ** 2))
                    delY = math.ceil(math.sqrt((MAJ * s) ** 2 + (MIN * c) ** 2))
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]
        
                    for deli in range(0, delX + 1):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                        
                    for deli in range(0, -delX - 1, -1):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                                    
                                                
                    if a[k] == 0:
                        xmin = max(0, xo - delX)
                        xmax = min(m - 1, xo + delX)
                        ymin = max(0, yo - delY)
                        ymax = min(n - 1, yo + delY)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                            
            MAJ = MAJ + 1.0              
            showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1
            
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.ThetaValues = Theta
    obj.Rvalues = Rval
    obj.Col = Col
    return obj
   
def Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_without_aspect_with_theta_with_adot(obj):
       
    # Grabbing data from the input Object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float R = obj.R
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef long long seq = obj.seq
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Thetafunc = obj.Thetafunc
    Adotfunc = obj.Adotfunc
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long deli, delj, MN, PN, PE, PW, PS, Io, win, xmin, xmax, ymin, ymax, dum, countim, i, j, k, xo, yo, delX, delY, MAJceil, MINceil, Iter
    cdef float MIN, theta, c, s, dr, tmp, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] Theta = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] MAJ = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] adotVal = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()
        
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)
      
    Iter = 1
    while True:
      
        if seq == 1 or seq == 2 or seq == 3:
            for k in range(0, p):
                I[X[k], Y[k]] = k + 1
                Col[X[k], Y[k], 2] = col[k, 0]
                Col[X[k], Y[k], 1] = col[k, 1]
                Col[X[k], Y[k], 0] = col[k, 2]
                Theta[k] = Thetafunc(X[k], Y[k], k + 1, 1, 1)
                adotVal[k] = Adotfunc(X[k], Y[k], k + 1, Theta[k], 1)
        else:
            for k in range(0, p):
                I[X[k], Y[k]] = k + 1
                Col[X[k], Y[k], 2] = col[k, 0]
                Col[X[k], Y[k], 1] = col[k, 1]
                Col[X[k], Y[k], 0] = col[k, 2]
                adotVal[k] = Adotfunc(X[k], Y[k], k + 1, 0, 1)
                Theta[k] = Thetafunc(X[k], Y[k], k + 1, 1, adotVal[k])
                   
               
                
        showriteframe(sa, sf, fd, 0, I, Col, p, Iter, framepause)
         
        countim = 0
        while cv2.countNonZero(a) > 0:
            countim = countim + 1
            adotValmax = 0
            for i in range(0, p):
                if a[i] == 1:
                    tmp = adotVal[i] 
                    if tmp > adotValmax:         
                        adotValmax = tmp        
                
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]
                    theta = Theta[k]
                        
                    c = math.cos(theta)
                    s = -math.sin(theta)
                                                       
                    dr = adotVal[k] / adotValmax
                    MAJ[k] = MAJ[k] + dr
                    MIN = MAJ[k] / R
                    MAJceil = math.ceil(MAJ[k])
                    MINceil = math.ceil(MIN)
                        
                        
                    delX = math.ceil(math.sqrt((MAJ[k] * c) ** 2 + (MIN * s) ** 2))
                    delY = math.ceil(math.sqrt((MAJ[k] * s) ** 2 + (MIN * c) ** 2))
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]
                        
                    # OUTWARD
        
                    for deli in range(0, delX + 1):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ[k]) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                        
                    for deli in range(0, -delX - 1, -1):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ[k]) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                        
                        
                    # INWARD
        
                    for deli in range(delX, -1, -1):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ[k]) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                        
                    for deli in range(-delX , 0):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ[k]) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
           
                    if a[k] == 0:
                        xmin = max(0, xo - delX)
                        xmax = min(m - 1, xo + delX)
                        ymin = max(0, yo - delY)
                        ymax = min(n - 1, yo + delY)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                
            showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                MAJ = np.zeros(p, dtype=np.float64)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1
            
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.ThetaValues = Theta
    obj.adotValues = adotVal
    obj.Col = Col
    return obj
   
   
def Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_with_aspect_without_theta_with_adot(obj):
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef long long seq = obj.seq
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Adotfunc = obj.Adotfunc
    Rfunc = obj.Rfunc
       
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long deli, delj, MN, PN, PE, PW, PS, Io, win, xmin, xmax, ymin, ymax, imin, imax, jmin, jmax, io, jo, go, found, dum, i, j, k, xo, yo, count, l1, l2, countim, MAJceil, MINceil, Iter
    cdef float adotValmax, tmp, MIN, dr, R, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int_t, ndim = 2] II = np.zeros((m, n), dtype=np.int)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] adotVal = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] RVal = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] MAJ = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()
   
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)
      
    Iter = 1
    while True:
       
        if seq == 1 or seq == 3 or seq == 4:
            for k in range(0, p):
                I[X[k], Y[k]] = k + 1
                Col[X[k], Y[k], 2] = col[k, 0]
                Col[X[k], Y[k], 1] = col[k, 1]
                Col[X[k], Y[k], 0] = col[k, 2]
                RVal[k] = Rfunc(X[k], Y[k], k + 1, 1, 0)
                adotVal[k] = Adotfunc(X[k], Y[k], k + 1, 0, RVal[k])
        else:
            for k in range(0, p):
                I[X[k], Y[k]] = k + 1
                Col[X[k], Y[k], 2] = col[k, 0]
                Col[X[k], Y[k], 1] = col[k, 1]
                Col[X[k], Y[k], 0] = col[k, 2]
                adotVal[k] = Adotfunc(X[k], Y[k], k + 1, 0, 1)
                RVal[k] = Rfunc(X[k], Y[k], k + 1, adotVal[k], 0)
                   
               
                   
                
        adotValmax = max(adotVal)
                
        showriteframe(sa, sf, fd, 0, I, Col, p, Iter, framepause)
        
        countim = 1
        while cv2.countNonZero(a) > 0:
            adotValmax = 0
            for i in range(0, p):
                if a[i] == 1:
                    tmp = adotVal[i] 
                    if tmp > adotValmax:         
                        adotValmax = tmp
                
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]
                        
                    dr = adotVal[k] / adotValmax
                    MAJ[k] = MAJ[k] + dr
                    R = RVal[k]
                    MIN = MAJ[k] / R
                    MAJceil = math.ceil(MAJ[k])
                    MINceil = math.ceil(MIN)
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]
                        
                    imin, imax, jmin, jmax = xo, xo, yo, yo
                    count = 0
                    for deli in range(0, MAJceil + 1):
                        for delj in range(0, MINceil + 1):                
                            if (deli / MAJ[k]) ** 2 + (delj / MIN) ** 2 <= 1:
                                for l1 in range(-1, 2, 2):
                                    for l2 in range(-1, 2, 2):
                                        i, j = xo + l1 * deli, yo + l2 * delj
                                        if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                            if I[i, j] == 0:
                                                II[i, j] = Io
                                                count = count + 1                                            
                                                if i > imax:
                                                    imax = i
                                                if i < imin:
                                                    imin = i
                                                if j > jmax:
                                                    jmax = j
                                                if j < jmin:
                                                    jmin = j
                                                    
                    if count > 0:                                              
                        i = imin
                        go = 1
                        found = 0
                        while go is True:
                            for j in range(jmin, jmax + 1, 1):
                                if II[i, j] == Io:
                                    if i > 0:
                                        if  I[i - 1, j] == Io:
                                            io, jo = i, j
                                            go = 0
                                            found = 1
                                            break
                                    if j > 0:
                                        if I[i, j - 1] == Io:
                                            io, jo = i, j
                                            go = 0
                                            found = 1
                                            break
                                                                     
                                    if j < n - 1:
                                        if I[i, j + 1] == Io:
                                            io, jo = i, j
                                            go = 0
                                            found = 1
                                            break
                                          
                                        
                                    if i < m - 1:
                                        if  I[i + 1, j] == Io:
                                            io, jo = i, j
                                            go = 0
                                            found = 1
                                            break
                            i = i + 1
                            if i == imax + 1:
                                go = 0
                                    
                        if found == 1:
                                
                            # OUTWARD
                            for i in range(io, imax + 1):
                                for j in range(jo, jmax + 1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            
                                for j in range(jo - 1, jmin - 1, -1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                
                                
                                
                            for i in range(io - 1, imin - 1, -1):
                                for j in range(jo, jmax + 1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                    
                                for j in range(jo - 1, jmin - 1, -1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                
                                
                                
                            # INWARD
                            for i in range(imax, io - 1, -1):
                                for j in range(jo, jmax + 1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                    
                                for j in range(jo - 1, jmin - 1, -1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                
                                
                                
                            for i in range(imin, io, -1):
                                for j in range(jo, jmax + 1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                    
                                for j in range(jo - 1, jmin - 1, -1):
                                    for dum in range(1):                        
                                        if II[i, j] == Io:
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                
                    if a[k] == 0:
                        xmin = max(0, xo - MAJceil)
                        xmax = min(m - 1, xo + MAJceil)
                        ymin = max(0, yo - MINceil)
                        ymax = min(n - 1, yo + MINceil)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                            
                            
                            
                                        
            showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
            countim = countim + 1
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                MAJ = np.zeros(p, dtype=np.float64)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1
           
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.RVal = RVal
    obj.adotVal = adotVal
    obj.Col = Col
    return obj
   
def Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_with_aspect_with_theta_with_adot(obj):
       
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef int seq = obj.seq
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Adotfunc = obj.Adotfunc
    Thetafunc = obj.Thetafunc
    Rfunc = obj.Rfunc
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long deli, delj, MN, PN, PE, PW, PS, Io, win, xmin, xmax, ymin, ymax, dum, countim, i, j, k, xo, yo, delX, delY, MAJceil, MINceil, Iter
    cdef float MIN, theta, c, s, dr, tmp, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] Theta = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] MAJ = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] adotVal = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Rval = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((p, 3)) 
    setwindows(sa);tic = time.time()
       
        
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)
      
    Iter = 1
      
    while True:
      
        if seq == 1:
            for k in range(0, p):
                I[X[k], Y[k]] = k + 1
                Col[X[k], Y[k], 2] = col[k, 0]
                Col[X[k], Y[k], 1] = col[k, 1]
                Col[X[k], Y[k], 0] = col[k, 2]
                Theta[k] = Thetafunc(X[k], Y[k], k + 1, 1, 1)
                Rval[k] = Rfunc(X[k], Y[k], k + 1, 1, Theta[k])
                adotVal[k] = Adotfunc(X[k], Y[k], k + 1, Theta[k], Rval[k])
               
        elif seq == 2:
            for k in range(0, p):
                I[X[k], Y[k]] = k + 1
                Col[X[k], Y[k], 2] = col[k, 0]
                Col[X[k], Y[k], 1] = col[k, 1]
                Col[X[k], Y[k], 0] = col[k, 2]
                Theta[k] = Thetafunc(X[k], Y[k], k + 1, 1, 1)
                adotVal[k] = Adotfunc(X[k], Y[k], k + 1, Theta[k], 1)
                Rval[k] = Rfunc(X[k], Y[k], k + 1, adotVal[k], Theta[k])
           
        elif seq == 3:
            for k in range(0, p):
                I[X[k], Y[k]] = k + 1
                Col[X[k], Y[k], 2] = col[k, 0]
                Col[X[k], Y[k], 1] = col[k, 1]
                Col[X[k], Y[k], 0] = col[k, 2]
                Rval[k] = Rfunc(X[k], Y[k], k + 1, 1, 0)
                Theta[k] = Thetafunc(X[k], Y[k], k + 1, Rval[k], 1)
                adotVal[k] = Adotfunc(X[k], Y[k], k + 1, Theta[k], Rval[k])
               
          
        elif seq == 4:
            for k in range(0, p):
                I[X[k], Y[k]] = k + 1
                Col[X[k], Y[k], 2] = col[k, 0]
                Col[X[k], Y[k], 1] = col[k, 1]
                Col[X[k], Y[k], 0] = col[k, 2]
                Rval[k] = Rfunc(X[k], Y[k], k + 1, 1, 0)
                adotVal[k] = Adotfunc(X[k], Y[k], k + 1, 0, Rval[k])
                Theta[k] = Thetafunc(X[k], Y[k], k + 1, Rval[k], adotVal[k])
                      
         
        elif seq == 5:
            for k in range(0, p):
                I[X[k], Y[k]] = k + 1
                Col[X[k], Y[k], 2] = col[k, 0]
                Col[X[k], Y[k], 1] = col[k, 1]
                Col[X[k], Y[k], 0] = col[k, 2]
                adotVal[k] = Adotfunc(X[k], Y[k], k + 1, 0, 1)
                Theta[k] = Thetafunc(X[k], Y[k], k + 1, 1, adotVal[k])
                Rval[k] = Rfunc(X[k], Y[k], k + 1, adotVal[k], Theta[k])
          
        elif seq == 6:
            for k in range(0, p):
                I[X[k], Y[k]] = k + 1
                Col[X[k], Y[k], 2] = col[k, 0]
                Col[X[k], Y[k], 1] = col[k, 1]
                Col[X[k], Y[k], 0] = col[k, 2]
                adotVal[k] = Adotfunc(X[k], Y[k], k + 1, 0, 1)
                Rval[k] = Rfunc(X[k], Y[k], k + 1, adotVal[k], 0)
                Theta[k] = Thetafunc(X[k], Y[k], k + 1, Rval[k], adotVal[k])
               
        showriteframe(sa, sf, fd, 0, I, Col, p, Iter, framepause)
         
        countim = 0
        while cv2.countNonZero(a) > 0:
            countim = countim + 1
            adotValmax = 0
            for i in range(0, p):
                if a[i] == 1:
                    tmp = adotVal[i] 
                    if tmp > adotValmax:         
                        adotValmax = tmp        
                
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]
                    theta = Theta[k]
                        
                    c = math.cos(theta)
                    s = -math.sin(theta)
                                                       
                    dr = adotVal[k] / adotValmax
                    MAJ[k] = MAJ[k] + dr
                    MIN = MAJ[k] / Rval[k]
                    MAJceil = math.ceil(MAJ[k])
                    MINceil = math.ceil(MIN)
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]
                      
                    delX = math.ceil(math.sqrt((MAJ[k] * c) ** 2 + (MIN * s) ** 2))
                    delY = math.ceil(math.sqrt((MAJ[k] * s) ** 2 + (MIN * c) ** 2))
                        
                        
                        
                    # OUTWARD
        
                    for deli in range(0, delX + 1):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ[k]) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                        
                    for deli in range(0, -delX - 1, -1):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ[k]) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                        
                        
                    # INWARD
        
                    for deli in range(delX, -1, -1):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ[k]) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                        
                    for deli in range(-delX , 0):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ[k]) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                        
                                    
                                                
                    if a[k] == 0:
                        xmin = max(0, xo - delX)
                        xmax = min(m - 1, xo + delX)
                        ymin = max(0, yo - delY)
                        ymax = min(n - 1, yo + delY)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                
            showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                MAJ = np.zeros(p, dtype=np.float64)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1
            
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.RVal = Rval
    obj.adotValues = adotVal
    obj.ThetaValues = Theta
    obj.Col = Col
    return obj
   
    
def Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_without_aspect_without_theta_without_adot(obj):
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float fstop = obj.fstop
    cdef long long MN = m * n 
    cdef float R = obj.R
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause 
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Ndot = obj.Ndot
    Gt = obj.Gt       
    fd = obj.fd
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long p, deli, delj, PN, PE, PW, PS, Io, count, dp, nnz, rad, countim, w, xt, yt, i, j, k, xo, yo, MAJceil, MINceil, Iter
    cdef float dt, f, t, tmp, Min, Maj, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 1] MAJ = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)    
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int_t, ndim = 1] a = np.ones(MN, dtype=np.int)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time()
   
    t = 0
    p = 0
    dp = 0
    countim = 0
    nnz = 0
    Iter = 1
    while nnz < MN:
        f = nnz / MN
        dt = 1.0 / Gt(t)
        dp = int((1 - f) * Ndot(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                count = 0
                while count < dp:
                    xt = np.random.randint(0, m - 1)
                    yt = np.random.randint(0, n - 1)
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        MAJ[p + count] = 1
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                p = p + dp
        else:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                w = 0
                count = 0
                while count < dp:
                    xt = xtrial[w]
                    yt = ytrial[w]
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        MAJ[p + count] = 1
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                    if w < dp - 1:
                        w = w + 1                    
                    else:
                        xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                        w = 0
                p = p + dp 
               
          
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
   
        for k in range(0, p):
            if a[k] == 1:
                a[k] = 0
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]
                Maj = MAJ[k]
                Min = Maj / R
                MAJceil = math.ceil(Maj)
                MINceil = math.ceil(Min)
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]
                   
   
                for deli in range(MAJceil + 1):
                    for delj in range(MINceil + 1):
                        if (deli / Maj) ** 2 + (delj / Min) ** 2 < 1:
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if PN == Io or PW == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
   
                            i, j = xo - deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                    if j > 0:
                                        PW = I[i, j - 1]
                                    else:
                                        PW = 0
                                    if PW == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
   
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i < m - 1:
                                        PS = I[i + 1, j]
                                    else:
                                        PS = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if PE == Io or PS == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
   
                            i, j = xo + deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    if i > 0:
                                        PN = I[i - 1, j]
                                    else:
                                        PN = 0
                                    if j < n - 1:
                                        PE = I[i, j + 1]
                                    else:
                                        PE = 0
                                    if PN == Io or PE == Io:
                                        I[i, j] = Io
                                        Col[i, j, 2] = red
                                        Col[i, j, 1] = green 
                                        Col[i, j, 0] = blue
                                        a[k] = 1
                   
                if a[k] == 0:
                    xmin = max(0, xo - MAJceil)
                    xmax = min(m - 1, xo + MAJceil)
                    ymin = max(0, yo - MINceil)
                    ymax = min(n - 1, yo + MINceil)
                    a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                   
                   
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        for i in range(p):
            MAJ[i] = MAJ[i] + 1
        nnz = cv2.countNonZero(I)
       
    obj.p = p
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.Col = Col
    return obj
   
   
def Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_without_aspect_with_theta_without_adot(obj):
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float fstop = obj.fstop
    cdef long long MN = m * n 
    cdef float R = obj.R
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause 
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Ndot = obj.Ndot
    Gt = obj.Gt       
    fd = obj.fd
    Thetafunc = obj.Thetafunc
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long p, deli, delj, PN, PE, PW, PS, Io, count, dp, nnz, rad, countim, w, xt, yt, i, j, k, xo, yo, delX, delY, xmin, xmax, ymin, ymax, dum, Iter
    cdef float dt, f, t, tmp, Min, Maj, theta, c, s, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 1] MAJ = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)    
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int_t, ndim = 1] a = np.ones(MN, dtype=np.int)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] Theta = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time()
   
       
    
    t = 0
    p = 0
    dp = 0
    countim = 0
    nnz = 0
    Iter = 1 
    while nnz < MN:
        f = nnz / MN
        dt = 1.0 / Gt(t)
        dp = int((1 - f) * Ndot(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                count = 0
                while count < dp:
                    xt = np.random.randint(0, m - 1)
                    yt = np.random.randint(0, n - 1)
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        MAJ[p + count] = 1
                        Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, 1)
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                p = p + dp
        else:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                w = 0
                count = 0
                while count < dp:
                    xt = xtrial[w]
                    yt = ytrial[w]
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        MAJ[p + count] = 1
                        Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1 , 1)
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                    if w < dp - 1:
                        w = w + 1                    
                    else:
                        xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                        w = 0
                p = p + dp
               
          
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
   
        for k in range(0, p):
            if a[k] == 1:
                a[k] = 0
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]
                theta = Theta[k]
                c = math.cos(theta)
                s = -math.sin(theta)                    
                Maj = MAJ[k]
                Min = Maj / R
                delX = math.ceil(math.sqrt((Maj * c) ** 2 + (Min * s) ** 2))
                delY = math.ceil(math.sqrt((Maj * s) ** 2 + (Min * c) ** 2))
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]
   
                for deli in range(0, delX + 1):
                    for delj in range(0, delY + 1):
                        if ((deli * c - delj * s) / Maj) ** 2 + ((deli * s + delj * c) / Min) ** 2 < 1:
                                
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                
                                
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                   
                for deli in range(0, -delX - 1, -1):
                    for delj in range(0, delY + 1):
                        if ((deli * c - delj * s) / Maj) ** 2 + ((deli * s + delj * c) / Min) ** 2 < 1:
                                
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                
                                
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                   
                   
                   
                if a[k] == 0:
                    xmin = max(0, xo - delX)
                    xmax = min(m - 1, xo + delX)
                    ymin = max(0, yo - delY)
                    ymax = min(n - 1, yo + delY)
                    a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                   
                   
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        for i in range(p):
            MAJ[i] = MAJ[i] + 1
        nnz = cv2.countNonZero(I)
     
    obj.p = p
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.Col = Col
    return obj
   
def Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_with_aspect_without_theta_without_adot(obj):
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float fstop = obj.fstop
    cdef long long MN = m * n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause 
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Ndot = obj.Ndot
    Gt = obj.Gt       
    fd = obj.fd
    Rfunc = obj.Rfunc
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long p, deli, delj, PN, PE, PW, PS, Io, count, dp, nnz, rad, countim, w, xt, yt, i, j, k, xo, yo, MAJceil, MINceil, Iter
    cdef float dt, f, t, tmp, Min, Maj, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 1] MAJ = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)    
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int_t, ndim = 1] a = np.ones(MN, dtype=np.int)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] Rvals = np.ones(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time()
    
       
    
    t = 0
    p = 0
    dp = 0
    countim = 0
    nnz = 0
    Iter = 1
    if pdelNxy == []:
   
        while nnz < MN:
            f = nnz / MN
            dt = 1.0 / Gt(t)
            dp = int((1 - f) * Ndot(t) * dt)
            t = t + dt
            if pdelNxy == []:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Rvals[p + count] = Rfunc(xt, yt, p + count + 1, 1, 0)
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
            else:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Rvals[p + count] = Rfunc(xt, yt, p + count + 1, 1, 0)
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp    
                   
              
            showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
            countim = countim + 1
       
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    Io = I[xo, yo]
                    Maj = MAJ[k]
                    Min = Maj / Rvals[k]
                    MAJceil = math.ceil(Maj)
                    MINceil = math.ceil(Min)
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]
                      
                    for deli in range(MAJceil + 1):
                        for delj in range(MINceil + 1):
                            if (deli / Maj) ** 2 + (delj / Min) ** 2 < 1:
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if PN == Io or PW == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
       
                                i, j = xo - deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if j > 0:
                                            PW = I[i, j - 1]
                                        else:
                                            PW = 0
                                        if PW == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
       
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        if i < m - 1:
                                            PS = I[i + 1, j]
                                        else:
                                            PS = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if PE == Io or PS == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
       
                                i, j = xo + deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        if i > 0:
                                            PN = I[i - 1, j]
                                        else:
                                            PN = 0
                                        if j < n - 1:
                                            PE = I[i, j + 1]
                                        else:
                                            PE = 0
                                        if PN == Io or PE == Io:
                                            I[i, j] = Io
                                            Col[i, j, 2] = red
                                            Col[i, j, 1] = green 
                                            Col[i, j, 0] = blue
                                            a[k] = 1
                       
                    if a[k] == 0:
                        xmin = max(0, xo - MAJceil)
                        xmax = min(m - 1, xo + MAJceil)
                        ymin = max(0, yo - MINceil)
                        ymax = min(n - 1, yo + MINceil)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                       
                       
            showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
            for i in range(p):
                MAJ[i] = MAJ[i] + 1
            nnz = cv2.countNonZero(I)
   
    obj.p = p
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.Col = Col
    return obj
       
   
       
def Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_without_aspect_without_theta_with_adot(obj):
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float R = obj.R
    cdef float fstop = obj.fstop
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef long long MN = m * n
    fd = obj.fd
    Ndot = obj.Ndot
    Gt = obj.Gt 
    pdelNxy = obj.pdelNxy
    Adotfunc = obj.Adotfunc
       
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long deli, delj, PN, PE, PW, PS, Io, win, xmin, xmax, ymin, ymax, imin, imax, jmin, jmax, io, jo, go, found, dum, i, j, k, xo, yo, count, l1, l2, countim, MAJceil, MINceil, nnz, xt, yt, dp, p, Iter
    cdef float adotValmax, tmp, MIN, dr, theta, c, s, t, dt, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int_t, ndim = 1] a = np.ones(MN, dtype=np.int)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] adotVal = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim = 2] II = np.zeros((m, n), dtype=np.int)
    cdef np.ndarray[np.float64_t, ndim = 1] MAJ = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time()
   
   
    countim = 1
    count = 0
    nnz = 0
    t = 0
    p = 0
    Iter = 1
    while nnz < MN:         
        f = nnz / MN
        dt = 1.0 / Gt(t)
        dp = int((1 - f) * Ndot(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                count = 0
                while count < dp:
                    xt = np.random.randint(0, m - 1)
                    yt = np.random.randint(0, n - 1)
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        MAJ[p + count] = 1
                        adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, 1)
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                p = p + dp
        else:
            if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                w = 0
                count = 0
                while count < dp:
                    xt = xtrial[w]
                    yt = ytrial[w]
                    if I[xt, yt] == 0:
                        I[xt, yt] = p + count + 1
                        X[p + count] = xt
                        Y[p + count] = yt
                        a[p + count] = 1
                        MAJ[p + count] = 1
                        adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, 1)
                        Col[xt, yt, 2] = col[p + count, 0]
                        Col[xt, yt, 1] = col[p + count, 1]
                        Col[xt, yt, 0] = col[p + count, 2]
                        count = count + 1
                    if w < dp - 1:
                        w = w + 1                    
                    else:
                        xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                        w = 0
                p = p + dp
               
          
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
           
        adotValmax = 0
        for i in range(0, p):
            if a[i] == 1:
                tmp = adotVal[i] 
                if tmp > adotValmax:         
                    adotValmax = tmp
            
        for k in range(0, p):
            if a[k] == 1:
                a[k] = 0
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]
                    
                dr = adotVal[k] / adotValmax
                MAJ[k] = MAJ[k] + dr
                MIN = MAJ[k] / R
                MAJceil = math.ceil(MAJ[k])
                MINceil = math.ceil(MIN)
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]
                    
                imin, imax, jmin, jmax = xo, xo, yo, yo
    
                for deli in range(0, MAJceil + 1):
                    for delj in range(0, MINceil + 1):                
                        if (deli / MAJ[k]) ** 2 + (delj / MIN) ** 2 <= 1:
                            for l1 in range(-1, 2, 2):
                                for l2 in range(-1, 2, 2):
                                    i, j = xo + l1 * deli, yo + l2 * delj
                                    if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                        if I[i, j] == 0:
                                            II[i, j] = Io
                                            count = count + 1                                            
                                            if i > imax:
                                                imax = i
                                            if i < imin:
                                                imin = i
                                            if j > jmax:
                                                jmax = j
                                            if j < jmin:
                                                jmin = j
                                                
                if count > 0:                                              
                    i = imin
                    go = 1
                    found = 0
                    while go is True:
                        for j in range(jmin, jmax + 1, 1):
                            if II[i, j] == Io:
                                if i > 0:
                                    if  I[i - 1, j] == Io:
                                        io, jo = i, j
                                        go = 0
                                        found = 1
                                        break
                                if j > 0:
                                    if I[i, j - 1] == Io:
                                        io, jo = i, j
                                        go = 0
                                        found = 1
                                        break
                                                                 
                                if j < n - 1:
                                    if I[i, j + 1] == Io:
                                        io, jo = i, j
                                        go = 0
                                        found = 1
                                        break
                                      
                                    
                                if i < m - 1:
                                    if  I[i + 1, j] == Io:
                                        io, jo = i, j
                                        go = 0
                                        found = 1
                                        break
                        i = i + 1
                        if i == imax + 1:
                            go = 0
                                
                    if found == 1:
                            
                        # OUTWARD
                        for i in range(io, imax + 1):
                            for j in range(jo, jmax + 1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        
                            for j in range(jo - 1, jmin - 1, -1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                            
                            
                            
                        for i in range(io - 1, imin - 1, -1):
                            for j in range(jo, jmax + 1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                
                            for j in range(jo - 1, jmin - 1, -1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                            
                            
                            
                        # INWARD
                        for i in range(imax, io - 1, -1):
                            for j in range(jo, jmax + 1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                
                            for j in range(jo - 1, jmin - 1, -1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                            
                            
                            
                        for i in range(imin, io, -1):
                            for j in range(jo, jmax + 1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                
                            for j in range(jo - 1, jmin - 1, -1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                            
                if a[k] == 0:
                    xmin = max(0, xo - MAJceil)
                    xmax = min(m - 1, xo + MAJceil)
                    ymin = max(0, yo - MINceil)
                    ymax = min(n - 1, yo + MINceil)
                    a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                        
                        
                        
                                    
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
        nnz = cv2.countNonZero(I)
      
    obj.p = p
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.adotVal = adotVal
    obj.Col = Col
    return obj
   
def Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_with_aspect_with_theta_without_adot(obj):
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float fstop = obj.fstop
    cdef long long MN = m * n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef long long seq = obj.seq 
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Ndot = obj.Ndot
    Gt = obj.Gt       
    fd = obj.fd
    Thetafunc = obj.Thetafunc
    Rfunc = obj.Rfunc
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long p, deli, delj, PN, PE, PW, PS, Io, count, dp, nnz, rad, countim, w, xt, yt, i, j, k, xo, yo, delX, delY, xmin, xmax, ymin, ymax, dum, Iter
    cdef float dt, f, t, tmp, Min, Maj, theta, c, s, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 1] MAJ = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)    
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int_t, ndim = 1] a = np.ones(MN, dtype=np.int)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] Theta = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Rvals = np.ones(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time()
    
       
    
    t = 0
    p = 0
    dp = 0
    countim = 0
    nnz = 0
    Iter = 1
    while nnz < MN:
        f = nnz / MN
        dt = 1.0 / Gt(t)
        dp = int((1 - f) * Ndot(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if seq == 3 or seq == 4 or seq == 6:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Rvals[p + count] = Rfunc(xt, yt, p + count + 1, 1, 0)
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, Rvals[p + count], 1)
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
            else:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, 1)
                            Rvals[p + count] = Rfunc(xt, yt, p + count + 1, 1, Theta[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
        else:
            if seq == 3 or seq == 4 or seq == 6:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Rvals[p + count] = Rfunc(xt, yt, p + count + 1, 1, 0)
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, Rvals[p + count], 1)
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]                                                        
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp
            else:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, 1)
                            Rvals[p + count] = Rfunc(xt, yt, p + count + 1, 1, Theta[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]                            
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp
            
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
   
        for k in range(0, p):
            if a[k] == 1:
                a[k] = 0
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]
                theta = Theta[k]
                c = math.cos(theta)
                s = -math.sin(theta)                    
                Maj = MAJ[k]
                Min = Maj / Rvals[k]
                delX = math.ceil(math.sqrt((Maj * c) ** 2 + (Min * s) ** 2))
                delY = math.ceil(math.sqrt((Maj * s) ** 2 + (Min * c) ** 2))
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]
   
                for deli in range(0, delX + 1):
                    for delj in range(0, delY + 1):
                        if ((deli * c - delj * s) / Maj) ** 2 + ((deli * s + delj * c) / Min) ** 2 < 1:
                                
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                
                                
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                   
                for deli in range(0, -delX - 1, -1):
                    for delj in range(0, delY + 1):
                        if ((deli * c - delj * s) / Maj) ** 2 + ((deli * s + delj * c) / Min) ** 2 < 1:
                                
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                
                                
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                   
                   
                   
                if a[k] == 0:
                    xmin = max(0, xo - delX)
                    xmax = min(m - 1, xo + delX)
                    ymin = max(0, yo - delY)
                    ymax = min(n - 1, yo + delY)
                    a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                   
                   
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        for i in range(p):
            MAJ[i] = MAJ[i] + 1
           
        nnz = cv2.countNonZero(I)
          
    obj.p = p
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.Col = Col
    return obj
   
def Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_with_aspect_without_theta_with_adot(obj):
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float fstop = obj.fstop
    cdef long long MN = m * n
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef long long seq = obj.seq
    fd = obj.fd
    Ndot = obj.Ndot
    Gt = obj.Gt 
    pdelNxy = obj.pdelNxy
    Adotfunc = obj.Adotfunc
    Rfunc = obj.Rfunc
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long deli, delj, PN, PE, PW, PS, Io, win, xmin, xmax, ymin, ymax, imin, imax, jmin, jmax, io, jo, go, found, dum, i, j, k, xo, yo, count, l1, l2, countim, MAJceil, MINceil, nnz, xt, yt, dp, p, w, Iter
    cdef float adotValmax, tmp, MIN, dr, theta, c, s, t, dt, red, green, blue
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int_t, ndim = 1] a = np.ones(MN, dtype=np.int)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] adotVal = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim = 2] II = np.zeros((m, n), dtype=np.int)
    cdef np.ndarray[np.float64_t, ndim = 1] MAJ = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Rval = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time()
   
    countim = 1
    count = 0
    nnz = 0
    t = 0
    p = 0
       
    Iter = 1
    while nnz < MN:
        f = nnz / MN
        dt = 1.0 / Gt(t)
        dp = int((1 - f) * Ndot(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if seq == 1 or seq == 3 or seq == 4:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, 1, 0)
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, Rval[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
            else:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, 1)
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, adotVal[p + count], 0)
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
        else:
            if seq == 1 or seq == 3 or seq == 4:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, 1, 0)
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, Rval[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp
            else:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, 1)
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, adotVal[p + count], 0)
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]                                    
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp
                   
               
          
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
           
        adotValmax = 0
        for i in range(0, p):
            if a[i] == 1:
                tmp = adotVal[i] 
                if tmp > adotValmax:         
                    adotValmax = tmp
            
        for k in range(0, p):
            if a[k] == 1:
                a[k] = 0
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]
                    
                dr = adotVal[k] / adotValmax
                MAJ[k] = MAJ[k] + dr
                MIN = MAJ[k] / Rval[k]
                MAJceil = math.ceil(MAJ[k])
                MINceil = math.ceil(MIN)
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]
                    
                imin, imax, jmin, jmax = xo, xo, yo, yo
    
                for deli in range(0, MAJceil + 1):
                    for delj in range(0, MINceil + 1):                
                        if (deli / MAJ[k]) ** 2 + (delj / MIN) ** 2 <= 1:
                            for l1 in range(-1, 2, 2):
                                for l2 in range(-1, 2, 2):
                                    i, j = xo + l1 * deli, yo + l2 * delj
                                    if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                        if I[i, j] == 0:
                                            II[i, j] = Io
                                            count = count + 1                                            
                                            if i > imax:
                                                imax = i
                                            if i < imin:
                                                imin = i
                                            if j > jmax:
                                                jmax = j
                                            if j < jmin:
                                                jmin = j
                                                
                if count > 0:                                              
                    i = imin
                    go = 1
                    found = 0
                    while go is True:
                        for j in range(jmin, jmax + 1, 1):
                            if II[i, j] == Io:
                                if i > 0:
                                    if  I[i - 1, j] == Io:
                                        io, jo = i, j
                                        go = 0
                                        found = 1
                                        break
                                if j > 0:
                                    if I[i, j - 1] == Io:
                                        io, jo = i, j
                                        go = 0
                                        found = 1
                                        break
                                                                 
                                if j < n - 1:
                                    if I[i, j + 1] == Io:
                                        io, jo = i, j
                                        go = 0
                                        found = 1
                                        break
                                      
                                    
                                if i < m - 1:
                                    if  I[i + 1, j] == Io:
                                        io, jo = i, j
                                        go = 0
                                        found = 1
                                        break
                        i = i + 1
                        if i == imax + 1:
                            go = 0
                                
                    if found == 1:
                            
                        # OUTWARD
                        for i in range(io, imax + 1):
                            for j in range(jo, jmax + 1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        
                            for j in range(jo - 1, jmin - 1, -1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                            
                            
                            
                        for i in range(io - 1, imin - 1, -1):
                            for j in range(jo, jmax + 1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                
                            for j in range(jo - 1, jmin - 1, -1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                            
                            
                            
                        # INWARD
                        for i in range(imax, io - 1, -1):
                            for j in range(jo, jmax + 1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                
                            for j in range(jo - 1, jmin - 1, -1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                            
                            
                            
                        for i in range(imin, io, -1):
                            for j in range(jo, jmax + 1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                
                            for j in range(jo - 1, jmin - 1, -1):
                                for dum in range(1):                        
                                    if II[i, j] == Io:
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                            
                if a[k] == 0:
                    xmin = max(0, xo - MAJceil)
                    xmax = min(m - 1, xo + MAJceil)
                    ymin = max(0, yo - MINceil)
                    ymax = min(n - 1, yo + MINceil)
                    a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                        
                        
                        
                                    
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
        nnz = cv2.countNonZero(I)
           
       
       
    obj.p = p
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.adotVal = adotVal
    obj.Col = Col
    return obj
   
def Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_without_aspect_with_theta_with_adot(obj):
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float fstop = obj.fstop
    cdef long long MN = m * n 
    cdef float R = obj.R
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause 
    cdef long long seq = obj.seq
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Ndot = obj.Ndot
    Gt = obj.Gt       
    fd = obj.fd
    Thetafunc = obj.Thetafunc
    Adotfunc = obj.Adotfunc
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long p, deli, delj, PN, PE, PW, PS, Io, count, dp, nnz, rad, countim, w, xt, yt, i, j, k, xo, yo, delX, delY, xmin, xmax, ymin, ymax, dum, Iter
    cdef float dt, f, t, tmp, Min, Maj, theta, c, s, dr, red, green, blue
    cdef np.ndarray[np.float64_t, ndim = 1] MAJ = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)    
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int_t, ndim = 1] a = np.ones(MN, dtype=np.int)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] Theta = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] adotVal = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time()
       
    
    t = 0
    p = 0
    dp = 0
    countim = 0
    nnz = 0
    Iter = 1
    while nnz < MN:
        f = nnz / MN
        dt = 1.0 / Gt(t)
        dp = int((1 - f) * Ndot(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if seq == 4 or seq == 5 or seq == 6:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, 1)
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, adotVal[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
            else:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, 1)
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, Theta[p + count], 1)
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
        else:
            if seq == 4 or seq == 5 or seq == 6:                           
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, 1)
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, adotVal[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp
            else:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, 1)
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, Theta[p + count], 1)
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp
   
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
           
           
        adotValmax = 0
        for i in range(0, p):
            if a[i] == 1:
                tmp = adotVal[i] 
                if tmp > adotValmax:         
                    adotValmax = tmp
   
        for k in range(0, p):
            if a[k] == 1:
                a[k] = 0
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]
                theta = Theta[k]
                c = math.cos(theta)
                s = -math.sin(theta)
                dr = adotVal[k] / adotValmax
                MAJ[k] = MAJ[k] + dr
                Maj = MAJ[k]
                Min = Maj / R
                delX = math.ceil(math.sqrt((Maj * c) ** 2 + (Min * s) ** 2))
                delY = math.ceil(math.sqrt((Maj * s) ** 2 + (Min * c) ** 2))
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]
   
                for deli in range(0, delX + 1):
                    for delj in range(0, delY + 1):
                        if ((deli * c - delj * s) / Maj) ** 2 + ((deli * s + delj * c) / Min) ** 2 < 1:
                                
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                
                                
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                   
                for deli in range(0, -delX - 1, -1):
                    for delj in range(0, delY + 1):
                        if ((deli * c - delj * s) / Maj) ** 2 + ((deli * s + delj * c) / Min) ** 2 < 1:
                                
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                
                                
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                   
                   
                   
                if a[k] == 0:
                    xmin = max(0, xo - delX)
                    xmax = min(m - 1, xo + delX)
                    ymin = max(0, yo - delY)
                    ymax = min(n - 1, yo + delY)
                    a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                   
                   
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        nnz = cv2.countNonZero(I)
          
           
    obj.p = p
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.Col = Col
    return obj
   
   
   
def Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_with_aspect_with_theta_with_adot(obj):
    # Grabbing data from the input object
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef long long MN = m * n 
    cdef float fstop = obj.fstop
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef long long seq = obj.seq 
    fd = obj.fd
    pdelNxy = obj.pdelNxy
    Ndot = obj.Ndot
    Gt = obj.Gt       
    fd = obj.fd
    Thetafunc = obj.Thetafunc
    Adotfunc = obj.Adotfunc
    Rfunc = obj.Rfunc
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long p, deli, delj, PN, PE, PW, PS, Io, count, dp, nnz, rad, countim, w, xt, yt, i, j, k, xo, yo, delX, delY, xmin, xmax, ymin, ymax, dum, Iter
    cdef float dt, f, t, tmp, Min, Maj, theta, c, s, dr, red, green, blue
    cdef np.ndarray[np.float64_t, ndim = 1] MAJ = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 1] xtrial = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] ytrial = np.zeros(MN, dtype=np.int64)    
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int_t, ndim = 1] a = np.ones(MN, dtype=np.int)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(MN, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 1] Theta = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] adotVal = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Rval = np.zeros(MN, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((MN, 3), dtype=np.float64)
    plantseed(myseed)
    col = np.random.random((MN, 3)) 
    setwindows(sa);tic = time.time()
   
    t = 0
    p = 0
    dp = 0
    countim = 0
    nnz = 0
    Iter = 1
    while nnz < MN:
        f = nnz / MN
        dt = 1.0 / Gt(t)
        dp = int((1 - f) * Ndot(t) * dt)
        t = t + dt
        if pdelNxy == []:
            if seq == 1:                    
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, 1)
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, 1, Theta[p + count])
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, Theta[p + count], Rval[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
               
            elif  seq == 2:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, 1)
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, Theta[p + count], 1)
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, adotVal[p + count], Theta[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
              
            elif  seq == 3:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, 1, 0)
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, Rval[p + count], 1)
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, Theta[p + count], Rval[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
               
            elif  seq == 4:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, 1, 0)
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, Rval[p + count])
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, Rval[p + count], adotVal[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
              
            elif  seq == 5:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, 1)
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, adotVal[p + count])
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, adotVal[p + count], Theta[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
              
            elif  seq == 6:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    count = 0
                    while count < dp:
                        xt = np.random.randint(0, m - 1)
                        yt = np.random.randint(0, n - 1)
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, 1)
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, adotVal[p + count], 0)
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, Rval[p + count], adotVal[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                    p = p + dp
          
        else:
            if seq == 1:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, 1)
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, 1, Theta[p + count])
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, Theta[p + count], Rval[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp
            elif seq == 2:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, 1)
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, Theta[p + count], 1)
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, adotVal[p + count], Theta[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp
            elif seq == 3:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, 1, 0)
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, Rval[p + count], 1)
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, Theta[p + count], Rval[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp
            elif seq == 4:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, 1, 0)
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, Rval[p + count])
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, Rval[p + count], adotVal[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp
            elif seq == 5:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, 1)
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, 1, adotVal[p + count])
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, adotVal[p + count], Theta[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp
            elif seq == 6:
                if dp > 0 and dp <= MN - nnz and nnz < fstop * MN:
                    xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                    w = 0
                    count = 0
                    while count < dp:
                        xt = xtrial[w]
                        yt = ytrial[w]
                        if I[xt, yt] == 0:
                            I[xt, yt] = p + count + 1
                            X[p + count] = xt
                            Y[p + count] = yt
                            a[p + count] = 1
                            MAJ[p + count] = 1
                            adotVal[p + count] = Adotfunc(xt, yt, p + count + 1, 0, 1)
                            Rval[p + count] = Rfunc(xt, yt, p + count + 1, adotVal[p + count], 0)
                            Theta[p + count] = Thetafunc(xt, yt, p + count + 1, Rval[p + count], adotVal[p + count])
                            Col[xt, yt, 2] = col[p + count, 0]
                            Col[xt, yt, 1] = col[p + count, 1]
                            Col[xt, yt, 0] = col[p + count, 2]
                            count = count + 1
                        if w < dp - 1:
                            w = w + 1                    
                        else:
                            xtrial, ytrial = met.MH2D(pdelNxy, 0, m - 1, 0, n - 1, dp, 1)
                            w = 0
                    p = p + dp
               
        
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        countim = countim + 1
           
           
        adotValmax = 0
        for i in range(0, p):
            if a[i] == 1:
                tmp = adotVal[i] 
                if tmp > adotValmax:         
                    adotValmax = tmp
   
        for k in range(0, p):
            if a[k] == 1:
                a[k] = 0
                xo = X[k]
                yo = Y[k]
                Io = I[xo, yo]
                theta = Theta[k]
                c = math.cos(theta)
                s = -math.sin(theta)
                dr = adotVal[k] / adotValmax
                MAJ[k] = MAJ[k] + dr
                Maj = MAJ[k]
                Min = Maj / Rval[k]
                delX = math.ceil(math.sqrt((Maj * c) ** 2 + (Min * s) ** 2))
                delY = math.ceil(math.sqrt((Maj * s) ** 2 + (Min * c) ** 2))
                red = col[k, 0]
                green = col[k, 1]
                blue = col[k, 2]
   
                for deli in range(0, delX + 1):
                    for delj in range(0, delY + 1):
                        if ((deli * c - delj * s) / Maj) ** 2 + ((deli * s + delj * c) / Min) ** 2 < 1:
                                
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                
                                
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                   
                for deli in range(0, -delX - 1, -1):
                    for delj in range(0, delY + 1):
                        if ((deli * c - delj * s) / Maj) ** 2 + ((deli * s + delj * c) / Min) ** 2 < 1:
                                
                            i, j = xo + deli, yo + delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                
                                
                            i, j = xo - deli, yo - delj
                            if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                if I[i, j] == 0:
                                    for dum in range(1):             
    
                                        if i > 0:
                                            if  I[i - 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                        if j > 0:
                                            if I[i, j - 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                                                         
                                        if j < n - 1:
                                            if I[i, j + 1] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                                              
                                            
                                        if i < m - 1:
                                            if  I[i + 1, j] == Io:
                                                I[i, j] = Io
                                                Col[i, j, 2] = red
                                                Col[i, j, 1] = green 
                                                Col[i, j, 0] = blue
                                                a[k] = 1
                                                break
                   
                   
                   
                if a[k] == 0:
                    xmin = max(0, xo - delX)
                    xmax = min(m - 1, xo + delX)
                    ymin = max(0, yo - delY)
                    ymax = min(n - 1, yo + delY)
                    a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                   
                   
        showriteframe(sa, sf, fd, countim, I, Col, p, Iter, framepause)
        nnz = cv2.countNonZero(I)
       
     
    obj.p = p
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = np.delete(X, np.arange(p, MN))
    obj.Y = np.delete(Y, np.arange(p, MN))
    obj.I = I
    obj.Col = Col
    return obj
  
  
def Evolve_2D_Anisotropic_SiteSaturated_Generic_TextureBased_Ellipsoidal_without_ab_without_ac_without_adot(obj):
    # Grabbing data from the input object
    cdef long long p = obj.p
    cdef long long sf = obj.sf
    cdef long long sa = obj.sa
    cdef long long m = obj.m
    cdef long long n = obj.n
    cdef float Rab = obj.Rab
    cdef float Rac = obj.Rac
    cdef np.ndarray[np.float64_t, ndim = 3] EUL = obj.EUL
    cdef long long myseed = obj.myseed
    cdef int framepause = obj.framepause
    cdef int asy = obj.asy
    cdef int labelsorted = obj.labelsorted
    fd = obj.fd
    pdelNxy = obj.pdelNxy
       
    # Declaring other variables
    cdef double tic, toc
    cdef long long i, j, deli, delj, xo, yo, PN, PE, PW, PS , k, Io, countim, delX, delY, aAXIS, Iter 
    cdef float bAXIS, cAXIS, red, green, blue, theta, cx, cy, MAJ, MIN, A, H, B, c, s, c2, s2, Rab2, Rac2, hsc
    cdef np.ndarray[np.int64_t, ndim = 2] I = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 2] Iold = np.zeros((m, n), dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim = 1] a = np.ones(p, dtype=np.int8)
    cdef np.ndarray[np.int64_t, ndim = 1] X = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim = 1] Y = np.zeros(p, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim = 2] col = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Col = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Colcanvas = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] major2d = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] minor2d = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Theta = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] Rot = np.zeros((p, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 4] ROT = np.zeros((m, n, 3, 3), dtype=np.float64)
    
    ROT = conv.zxz_to_rm(EUL.reshape(m * n, 3)).reshape(m, n, 3, 3)
      
    plantseed(myseed)
    Colcanvas = COLOR.EULToCol(EUL) 
    setwindows(sa);tic = time.time()
            
    if pdelNxy == []:
        [X, Y] = randindex2D(m, n, p)
    else:
        X, Y = met.MH2D(pdelNxy, 0, m, 0, n, p, 1)        
            
    if labelsorted == 1:
        X, Y = labelsort(n, X, Y)        
      
      
    Iter = 1
    Rab2 = Rab ** 2
    Rac2 = Rac ** 2

    
    while True:
        aAXIS = 0
        for k in range(0, p):
            I[X[k], Y[k]] = k + 1
            Rot[k, 0, 0] = ROT[X[k], Y[k], 0, 0]
            Rot[k, 0, 1] = ROT[X[k], Y[k], 0, 1]
            Rot[k, 0, 2] = ROT[X[k], Y[k], 0, 2]
            Rot[k, 1, 0] = ROT[X[k], Y[k], 1, 0]
            Rot[k, 1, 1] = ROT[X[k], Y[k], 1, 1]
            Rot[k, 1, 2] = ROT[X[k], Y[k], 1, 2]
            Rot[k, 2, 0] = ROT[X[k], Y[k], 2, 0]
            Rot[k, 2, 1] = ROT[X[k], Y[k], 2, 1]
            Rot[k, 2, 2] = ROT[X[k], Y[k], 2, 2]
            col[k, 0] = Colcanvas[X[k], Y[k], 0]
            col[k, 1] = Colcanvas[X[k], Y[k], 1]
            col[k, 2] = Colcanvas[X[k], Y[k], 2]
            Col[X[k], Y[k], 0] = col[k, 0]
            Col[X[k], Y[k], 1] = col[k, 1]
            Col[X[k], Y[k], 2] = col[k, 2]
               
        showriteframe(sa, sf, fd, aAXIS, I, Col, p, Iter, framepause)
           
        aAXIS = 1
          
        
        for k in range(p):
            A = Rot[k, 0, 0] ** 2 + Rab2 * Rot[k, 1, 0] ** 2 + Rac2 * Rot[k, 2, 0] ** 2
            B = Rot[k, 0, 1] ** 2 + Rab2 * Rot[k, 1, 1] ** 2 + Rac2 * Rot[k, 2, 1] ** 2
            H = Rot[k, 0, 0] * Rot[k, 0, 1] + Rab2 * Rot[k, 1, 0] * Rot[k, 1, 1] + Rac2 * Rot[k, 2, 0] * Rot[k, 2, 1]
            
            Theta[k] = 0.5 * math.atan2(2 * H, (A - B))
            c = math.cos(Theta[k])
            s = math.sin(Theta[k])
            c2 = c ** 2
            s2 = s ** 2
            hsc = 2 * H * s * c
            cx = A * c2 + B * s2 + hsc 
            cy = B * c2 + A * s2 - hsc
            
            if cx <= cy:
                major2d[k] = 1 / math.sqrt(cx)
                minor2d[k] = 1 / math.sqrt(cy)
            else:
                Theta[k] = pi / 2 + Theta[k]
                major2d[k] = 1 / math.sqrt(cy)
                minor2d[k] = 1 / math.sqrt(cx)
        
        while cv2.countNonZero(a) > 0:
            aAXIS = aAXIS + 1
            for k in range(0, p):
                if a[k] == 1:
                    a[k] = 0
                    xo = X[k]
                    yo = Y[k]
                    theta = Theta[k]
                    c = math.cos(theta)
                    s = -math.sin(theta)
                    Io = I[xo, yo]
                    MAJ = aAXIS * major2d[k]
                    MIN = aAXIS * minor2d[k]
                    delX = math.ceil(math.sqrt((MAJ * c) ** 2 + (MIN * s) ** 2))
                    delY = math.ceil(math.sqrt((MAJ * s) ** 2 + (MIN * c) ** 2))
                    red = col[k, 0]
                    green = col[k, 1]
                    blue = col[k, 2]
        
                    for deli in range(0, delX + 1):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                        
                    for deli in range(0, -delX - 1, -1):
                        for delj in range(0, delY + 1):
                            if ((deli * c - delj * s) / MAJ) ** 2 + ((deli * s + delj * c) / MIN) ** 2 <= 1:
                                    
                                i, j = xo + deli, yo + delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                    
                                    
                                i, j = xo - deli, yo - delj
                                if 0 <= i <= m - 1 and 0 <= j <= n - 1:
                                    if I[i, j] == 0:
                                        for dum in range(1):             
        
                                            if i > 0:
                                                if  I[i - 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                            if j > 0:
                                                if I[i, j - 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                                             
                                            if j < n - 1:
                                                if I[i, j + 1] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                                                  
                                                
                                            if i < m - 1:
                                                if  I[i + 1, j] == Io:
                                                    I[i, j] = Io
                                                    Col[i, j, 2] = red
                                                    Col[i, j, 1] = green 
                                                    Col[i, j, 0] = blue
                                                    a[k] = 1
                                                    break
                        
                                    
                                                
                    if a[k] == 0:
                        xmin = max(0, xo - delX)
                        xmax = min(m - 1, xo + delX)
                        ymin = max(0, yo - delY)
                        ymax = min(n - 1, yo + delY)
                        a[k] = checkalive(I, m, n, Io, xmin, xmax, ymin, ymax)
                
            showriteframe(sa, sf, fd, aAXIS, I, Col, p, Iter, framepause)
            aAXIS = aAXIS + 1
            bAXIS = aAXIS / Rab
            cAXIS = aAXIS / Rac
        if asy == 0:
            break
        else:
            if (I == Iold).all() == 1:
                break
            else:
                Iold = I
                I = np.zeros((m, n), dtype=np.int64)
                a = np.ones(p, dtype=np.int8)
                X , Y = morph.centroids(Iold, m, n, p)
                Iter = Iter + 1        
      
    toc = time.time()
    obj.exetime = toc - tic
    obj.X = X
    obj.Y = Y
    obj.I = I
    obj.Thetavalues = Theta
    obj.Col = Col
    return obj