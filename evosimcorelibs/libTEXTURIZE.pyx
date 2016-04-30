# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
from __future__ import division

from cmath import sqrt
from math import gamma, atan2
import math
import os
import random
import sys
import time

import cv2

import libCONVERT as conv
import libMETROPOLIS as met
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

def plantseed(myseed):
    if myseed > 0:
        np.random.seed(myseed)
        random.seed(myseed)

def Texture_URO_xyz(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int makecanvas = obj.makecanvas
    cdef double tic, toc 
    
    cdef int i
    cdef float th, ph, lm , sth, sph, cth, cph, clm, slm
    cdef np.ndarray[np.float64_t, ndim = 1] r1 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] lambdaa = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] theta = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] phi = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] V1 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] V2 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Xc = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Yc = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Zc = np.zeros(3, dtype=np.float64)
    
    # DESCRIPTORS OF ORIENTATIONS
    cdef np.ndarray[np.float64_t, ndim = 3] Rot = np.zeros((p, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Eul = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Axis = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Angle = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Nd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rv = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Q = np.zeros((p, 4), dtype=np.float64)
    
    plantseed(myseed)
    tic = time.time()

    r1 = np.random.uniform(0, 1, [p, ])
    theta = np.arccos(2 * r1 - 1)
    phi = np.random.uniform(0, 2 * np.pi, [p, ])
    lambdaa = np.random.uniform(0, 2 * pi, [p, ])
  
    for i in range(p):
        th = theta[i]
        ph = phi[i]
        lm = lambdaa[i]
        cth = cos(th)
        cph = cos(ph)
        clm = cos(lm)
        
        sth = sin(th)
        sph = sin(ph)
        slm = sin(lm)
        
        # CRYSTALLOGRAPHIC X AXIS
        Xc[0] = sth * cph
        Xc[1] = sth * sph
        Xc[2] = cth
        
        # ORTHOGONAL AUXILARY VECTORS PERPENDICULAR TO X AXIS
        V1[0] = sph
        V1[1] = -cph
        V1[2] = 0
        V2[0] = cph * cth
        V2[1] = sph * cth
        V2[2] = -sth
        
        # CRYSTALLOGRAPHIC Y AXIS
        Yc[0] = clm * V1[0] + slm * V2[0]
        Yc[1] = clm * V1[1] + slm * V2[1]
        Yc[2] = clm * V1[2] + slm * V2[2]
        
        # CRYSTALLOGRAPHIC Z AXIS
        
        Zc[0] = Xc[1] * Yc[2] - Yc[1] * Xc[2]
        Zc[1] = Xc[2] * Yc[0] - Yc[2] * Xc[0]
        Zc[2] = Xc[0] * Yc[1] - Yc[0] * Xc[1]
        
        # ROTATION MATRIX
        Rot[i, 0, 0] = Xc[0]
        Rot[i, 0, 1] = Xc[1]
        Rot[i, 0, 2] = Xc[2]
        Rot[i, 1, 0] = Yc[0]
        Rot[i, 1, 1] = Yc[1]
        Rot[i, 1, 2] = Yc[2]
        Rot[i, 2, 0] = Zc[0]
        Rot[i, 2, 1] = Zc[1]
        Rot[i, 2, 2] = Zc[2]
        
    Eul = conv.rm_to_zxz(Rot)
    
   
    if makecanvas == 0:
        
        obj.Eul = Eul
    else:
        obj.EUL = getcanvas(m, n, Eul)
            
    toc = time.time()
    obj.exetime = toc - tic 
    return obj
        



def Texture_URO_nxyz(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int makecanvas = obj.makecanvas
    cdef double tic, toc
    
    cdef int i
    cdef float th, ph, lm , sth, sph, cth, cph, clm, slm, coss, sinn, cossncap0, cossncap1, cossncap2
    
    cdef np.ndarray[np.float64_t, ndim = 1] r1 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] lambdaa = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] theta = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] phi = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] ncap = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] V1 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] V2 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Xp = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Yp = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Zp = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Xc = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Yc = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Zc = np.zeros(p, dtype=np.float64)
    
    # DESCRIPTORS OF ORIENTATIONS
    cdef np.ndarray[np.float64_t, ndim = 3] Rot = np.zeros((p, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Eul = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Axis = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Angle = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Nd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rv = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Q = np.zeros((p, 4), dtype=np.float64)
    
    
    plantseed(myseed)
    tic = time.time()
    
    r1 = np.random.uniform(0, 1, [p, ])
    theta = np.arccos(2 * r1 - 1)
    phi = np.random.uniform(0, 2 * pi, [p, ])
    lambdaa = np.random.uniform(0, 2 * pi, [p, ])
    coss = 1 / sqrt(3.0)
    sinn = sqrt(2.0 / 3.0)

   
    for i in range(p):
        th = theta[i]
        ph = phi[i]
        cth = cos(th)
        cph = cos(ph)
        sth = sin(th)
        sph = sin(ph)

        # CRYSTALLOGRAPHIC ncap AXIS
        ncap[0] = sth * cph
        ncap[1] = sth * sph
        ncap[2] = cth
        
        
        # ORTHOGONAL AUXILARY VECTORS PERPENDICULAR TO ncap AXIS
        V1[0] = sph
        V1[1] = -cph
        V1[2] = 0
        V2[0] = cph * cth
        V2[1] = sph * cth
        V2[2] = -sth
       
        # PROJECTION OF X AXIS PERPENDICULAR TO ncap
        lm = lambdaa[i]
        clm = cos(lm)
        slm = sin(lm)
        Xp[0] = clm * V1[0] + slm * V2[0]
        Xp[1] = clm * V1[1] + slm * V2[1]
        Xp[2] = clm * V1[2] + slm * V2[2]
       
        # PROJECTION OF Y AXIS PERPENDICULAR TO ncap
        lm = lm + tpibth
        clm = cos(lm)
        slm = sin(lm)
        Yp[0] = clm * V1[0] + slm * V2[0]
        Yp[1] = clm * V1[1] + slm * V2[1]
        Yp[2] = clm * V1[2] + slm * V2[2]
    
       
        # PROJECTION OF Z AXIS PERPENDICULAR TO ncap
        lm = lm + tpibth
        clm = cos(lm)
        slm = sin(lm)
        Zp[0] = clm * V1[0] + slm * V2[0]
        Zp[1] = clm * V1[1] + slm * V2[1]
        Zp[2] = clm * V1[2] + slm * V2[2]
        
        # CRYSTALLOGRAPHIC X AXIS
        cossncap0 = coss * ncap[0]
        cossncap1 = coss * ncap[1]
        cossncap2 = coss * ncap[2]
        
        Xc[0] = cossncap0 + sinn * Xp[0]
        Xc[1] = cossncap1 + sinn * Xp[1]
        Xc[2] = cossncap2 + sinn * Xp[2]
        
        # CRYSTALLOGRAPHIC Y AXIS
        Yc[0] = cossncap0 + sinn * Yp[0]
        Yc[1] = cossncap1 + sinn * Yp[1]
        Yc[2] = cossncap2 + sinn * Yp[2]
        
        # CRYSTALLOGRAPHIC Z AXIS
        Zc[0] = cossncap0 + sinn * Zp[0]
        Zc[1] = cossncap1 + sinn * Zp[1]
        Zc[2] = cossncap2 + sinn * Zp[2]
        
        # ROTATION MATRIX
        Rot[i, 0, 0] = Xc[0]
        Rot[i, 0, 1] = Xc[1]
        Rot[i, 0, 2] = Xc[2]
        Rot[i, 1, 0] = Yc[0]
        Rot[i, 1, 1] = Yc[1]
        Rot[i, 1, 2] = Yc[2]
        Rot[i, 2, 0] = Zc[0]
        Rot[i, 2, 1] = Zc[1]
        Rot[i, 2, 2] = Zc[2]
        
    Eul = conv.rm_to_zxz(Rot)
    
    if makecanvas == 0:
        
        obj.Eul = Eul
        obj.Axis = Axis
        obj.Angle = Angle
        obj.Nd = Nd
        obj.Rd = Rd
        obj.Rv = Rv
        obj.Q = Q
        
    else:
        obj.EUL = getcanvas(m, n, Eul)
            
    toc = time.time()
    obj.exetime = toc - tic 
    return obj
        
        
def Texture_URO_arvo(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int makecanvas = obj.makecanvas
    cdef double tic, toc
    
    cdef int i, j, k, l
    cdef float ph, lm , sph, cph, clm, slm, squ, sum, u    
    
    cdef np.ndarray[np.float64_t, ndim = 1] rsign = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] phi = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] U = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] lambdaa = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] V = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] R = np.zeros((3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] H = np.zeros((3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] EYE = np.zeros((3, 3), dtype=np.float64)
    EYE = np.eye(3)
    
    # DESCRIPTORS OF ORIENTATIONS
    cdef np.ndarray[np.float64_t, ndim = 3] Rot = np.zeros((p, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Eul = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Axis = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Angle = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Nd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rv = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Q = np.zeros((p, 4), dtype=np.float64)
    
    
    plantseed(myseed)
    tic = time.time()
    
    rsign = np.random.uniform(0, 1, [p, ])
    phi = np.random.uniform(0, 2 * pi, [p, ])
    U = np.random.uniform(0, 1, [p, ])
    lambdaa = np.random.uniform(0, 2 * pi, [p, ])
   
    for k in range(p):
        ph = phi[k]
        cph = cos(ph)
        sph = sin(ph)
        u = U[k]
        lm = lambdaa[k]
        squ = sqrt(u)
        clm = cos(lm)
        slm = sin(lm)
         

        # ROTAION MATRIX FOR UNIFORM RANDOM ROTATION ABOUT Z AXIS
        R[0, 0] = cph
        R[0, 1] = sph
        R[0, 2] = 0
        R[1, 0] = -sph
        R[1, 1] = cph
        R[1, 2] = 0
        R[2, 0] = 0
        R[2, 1] = 0
        R[2, 2] = 1
         
        # UNIFORM RANDOM AUXILARY VECTOR
        V[0] = squ * clm
        V[1] = -squ * slm
        V[2] = sqrt(1 - u)
        if rsign[k] < 0.5:
            V[2] = -V[2]
        else:
            pass

              
        # HOUSEHOLDER MATRIX
        for i in range(3):
            for j in range(3):
                H[i, j] = EYE[i, j] - 2 * V[i] * V[j]
        
                
        # ROTATION MATRIX
        for i in range(3):
            for j in range(3):
                sum = 0
                for l in range(3):
                    sum = sum - H[i, l] * R[l, j]
                Rot[k, i, j] = sum
    
    Eul = conv.rm_to_zxz(Rot)
    
    if makecanvas == 0:
        
        obj.Eul = Eul
   
    else:
        obj.EUL = getcanvas(m, n, Eul)
            
    toc = time.time()
    obj.exetime = toc - tic 
    return obj


def Texture_URO_mack(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int makecanvas = obj.makecanvas
    cdef double tic, toc
    
    cdef int k
    cdef float x1, x2, x3, v1, v2, v3, S, T, P, SQ
    cdef np.ndarray[np.float64_t, ndim = 1] X1 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] X2 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] X3 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] V1 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] V2 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] V3 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] EYE = np.zeros((3, 3), dtype=np.float64)
    EYE = np.eye(3)
    cdef np.ndarray[np.float64_t, ndim = 1] Xc = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Yc = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Zc = np.zeros(p, dtype=np.float64)
   
    # DESCRIPTORS OF ORIENTATIONS
    cdef np.ndarray[np.float64_t, ndim = 3] Rot = np.zeros((p, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Eul = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Axis = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Angle = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Nd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rv = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Q = np.zeros((p, 4), dtype=np.float64)

    
    plantseed(myseed)
    tic = time.time()

    X1, X2, X3 = np.random.multivariate_normal([0, 0, 0], EYE, p).T
    V1, V2, V3 = np.random.multivariate_normal([0, 0, 0], EYE, p).T
    
  
    for k in range(p):
        x1 = X1[k]
        x2 = X2[k]
        x3 = X3[k]
        v1 = V1[k]
        v2 = V2[k]
        v3 = V3[k]
        S = x1 ** 2 + x2 ** 2 + x3 ** 2
        T = v1 ** 2 + v2 ** 2 + v3 ** 2
        P = x1 * v1 + x2 * v2 + x3 * v3
      
        # CRYSTALLOGRAPHIC X AXIS
        Xc[0] = x1 / sqrt(S)
        Xc[1] = x2 / sqrt(S)
        Xc[2] = x3 / sqrt(S)
        
        
        # CRYSTALLOGRAPHIC Y AXIS
        SQ = sqrt(S * (S * T - P ** 2))
        Yc[0] = (S * v1 - P * x1) / SQ
        Yc[1] = (S * v2 - P * x2) / SQ
        Yc[2] = (S * v3 - P * x3) / SQ
        
              
        # CRYSTALLOGRAPHIC Z AXIS        
        Zc[0] = Xc[1] * Yc[2] - Yc[1] * Xc[2]
        Zc[1] = Xc[2] * Yc[0] - Yc[2] * Xc[0]
        Zc[2] = Xc[0] * Yc[1] - Yc[0] * Xc[1]
        
        # ROTATION MATRIX
        Rot[k, 0, 0] = Xc[0]
        Rot[k, 0, 1] = Xc[1]
        Rot[k, 0, 2] = Xc[2]
        Rot[k, 1, 0] = Yc[0]
        Rot[k, 1, 1] = Yc[1]
        Rot[k, 1, 2] = Yc[2]
        Rot[k, 2, 0] = Zc[0]
        Rot[k, 2, 1] = Zc[1]
        Rot[k, 2, 2] = Zc[2]
        
    Eul = conv.rm_to_zxz(Rot)
    
    if makecanvas == 0:
        
        obj.Eul = Eul

    else:
        obj.EUL = getcanvas(m, n, Eul)
            
    toc = time.time()
    obj.exetime = toc - tic 
    return obj


def Texture_URO_sm1(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int makecanvas = obj.makecanvas
    cdef double tic, toc
    
    cdef int k
    cdef float u0, u1, u2, c1, s1, c2, s2, r1, r2, gamma1, gamma2
    
    cdef np.ndarray[np.float64_t, ndim = 1] U0 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] U1 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] U2 = np.zeros(p, dtype=np.float64)

    # DESCRIPTORS OF ORIENTATIONS
    cdef np.ndarray[np.float64_t, ndim = 3] Rot = np.zeros((p, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Eul = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Axis = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Angle = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Nd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rv = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Q = np.zeros((p, 4), dtype=np.float64)
    
    plantseed(myseed)
    tic = time.time()

    U0 = np.random.uniform(0, 1, [p, ])
    U1 = np.random.uniform(0, 1, [p, ])
    U2 = np.random.uniform(0, 1, [p, ])
  
    for k in range(p):
        u0 = U0[k]
        u1 = U1[k]
        u2 = U2[k]
        gamma1 = 2 * pi * u1
        gamma2 = 2 * pi * u2
        r1 = sqrt(1 - u0)
        r2 = sqrt(u0)
        c1 = cos(gamma1)
        s1 = sin(gamma1)
        c2 = cos(gamma2)
        s2 = sin(gamma2)
            
        Q[k, 0] = c2 * r2
        Q[k, 1] = s1 * r1
        Q[k, 2] = c1 * r1
        Q[k, 3] = s2 * r2
    
    Axis, Angle = conv.qn_to_aap(Q)    
    Rot = conv.aap_to_rm(Axis, Angle)
    Eul = conv.rm_to_zxz(Rot)
    
    if makecanvas == 0:
        
        obj.Eul = Eul
       
    else:
        obj.EUL = getcanvas(m, n, Eul)
            
    toc = time.time()
    obj.exetime = toc - tic 
    return obj


def Texture_URO_sm2(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int makecanvas = obj.makecanvas
    cdef double tic, toc
    
    cdef int k
    cdef float beta, theta, phi, cb, sb, st, cp, sp , ct
    
    cdef np.ndarray[np.float64_t, ndim = 1] r1 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Beta = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Theta = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Phi = np.zeros(p, dtype=np.float64)

    # DESCRIPTORS OF ORIENTATIONS
    cdef np.ndarray[np.float64_t, ndim = 3] Rot = np.zeros((p, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Eul = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Axis = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Angle = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Nd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rv = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Q = np.zeros((p, 4), dtype=np.float64)
    
    plantseed(myseed)
    tic = time.time()

    Beta = ProbSinSquare(0, pi, 1, p)
    r1 = np.random.uniform(0, 1, [p, ])
    Theta = np.arccos(2 * r1 - 1)
    Phi = np.random.uniform(0, 2 * pi, [p, ])
  
    for k in range(p):
        beta = Beta[k]
        theta = Theta[k]
        phi = Phi[k]
        cb = cos(beta)
        sb = sin(beta)
        st = sin(theta)
        cp = cos(phi)
        sp = sin(phi)
        ct = cos(theta)
            
        Q[k, 0] = cb
        Q[k, 1] = sb * st * cp
        Q[k, 2] = sb * st * sp
        Q[k, 3] = sb * ct
    
    Axis, Angle = conv.qn_to_aap(Q)    
    Rot = conv.aap_to_rm(Axis, Angle)
    Eul = conv.rm_to_zxz(Rot)
  

    
    if makecanvas == 0:
        
        obj.Eul = Eul
       
    else:
        obj.EUL = getcanvas(m, n, Eul)
            
    toc = time.time()
    obj.exetime = toc - tic 
    return obj

def Texture_URO_miles(obj):  # AXIS ANGLE PAIRS
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int makecanvas = obj.makecanvas
    cdef double tic, toc
    
    cdef int k
    cdef float st, sp, ct, cp, theta, phi
    
    cdef np.ndarray[np.float64_t, ndim = 1] r1 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Theta = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Phi = np.zeros(p, dtype=np.float64)

    # DESCRIPTORS OF ORIENTATIONS    
    cdef np.ndarray[np.float64_t, ndim = 3] Rot = np.zeros((p, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Eul = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Axis = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Angle = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Nd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rv = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Q = np.zeros((p, 4), dtype=np.float64)
    
    plantseed(myseed)
    tic = time.time()

    Angle = ProbSinSquare(0, pi, 2, p)
    r1 = np.random.uniform(0, 1, [p, ])
    Theta = np.arccos(2 * r1 - 1)
    Phi = np.random.uniform(0, 2 * pi, [p, ])
  
    for k in range(p):
        theta = Theta[k]
        phi = Phi[k]
        st = sin(theta)
        cp = cos(phi)
        sp = sin(phi)
        ct = cos(theta)
        
        Axis[k, 0] = st * cp
        Axis[k, 1] = st * sp
        Axis[k, 2] = ct 
        
    Rot = conv.aap_to_rm(Axis, Angle)
    Eul = conv.rm_to_zxz(Rot)
   

    
    if makecanvas == 0:
        
        obj.Eul = Eul
    else:
        obj.EUL = getcanvas(m, n, Eul)
            
    toc = time.time()
    obj.exetime = toc - tic 
    return obj


def Texture_URO_mur(obj):  # EULER TRIPLET BASED
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int makecanvas = obj.makecanvas
    cdef double tic, toc
    
    
    cdef np.ndarray[np.float64_t, ndim = 1] r1 = np.zeros(p, dtype=np.float64)

    # DESCRIPTORS OF ORIENTATIONS  
    cdef np.ndarray[np.float64_t, ndim = 3] Rot = np.zeros((p, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Eul = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Axis = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Angle = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Nd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rv = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Q = np.zeros((p, 4), dtype=np.float64)
    
    plantseed(myseed)
    tic = time.time()


    Eul[0:, 0] = np.random.uniform(0, 2 * pi, [p, ])
    r1 = np.random.uniform(0, 1, [p, ])
    Eul[0:, 1] = np.arccos(2 * r1 - 1)
    Eul[0:, 2] = np.random.uniform(0, 2 * pi, [p, ])
   
    Rot = conv.zxz_to_rm(Eul)
    
    if makecanvas == 0:
        
        obj.Eul = Eul
     
    else:
        obj.EUL = getcanvas(m, n, Eul)
        

    toc = time.time()
    obj.exetime = toc - tic 
    return obj


def Texture_Fiber(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int makecanvas = obj.makecanvas
    cdef double tic, toc
    
    cdef float xs = obj.xs
    cdef float ys = obj.ys
    cdef float zs = obj.zs
    cdef float xc = obj.xc
    cdef float yc = obj.yc
    cdef float zc = obj.zc
    cdef float thetamax = obj.thetamax
    cdef float thetamin = obj.thetamin
    axisdist = obj.axisdist
    
    cdef float normC 
    cdef float normS

    cdef int i, k
    cdef float th, ph, lm , sth, sph, cth, cph, clm, slm, STH, SPH, CTH, CPH, sthcph, sthsph, a
    cdef float alpha, beta, gamma, den1, den2, den3, den
    cdef np.ndarray[np.float64_t, ndim = 1] Lambdaa = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Phi = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] V1 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] V2 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] X1C = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] X2C = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] X3C = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] X1S = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] X2S = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] X3S = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] aux1 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] aux2 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] aux3 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Aux1 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Aux2 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Aux3 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] PHI1 = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] PHI2 = np.zeros(p, dtype=np.float64)

    # DESCRIPTORS OF ORIENTATIONS  
    cdef np.ndarray[np.float64_t, ndim = 3] Rot = np.zeros((p, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Eul = np.zeros((p, 3), dtype=np.float64)
    
    
    plantseed(myseed)
    tic = time.time()
    normC = sqrt(xc ** 2 + yc ** 2 + zc ** 2)
    normS = sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    xs = xs / normS; ys = ys / normS; zs = zs / normS;
    xc = xc / normC; yc = yc / normC; zc = zc / normC;

    
    # CONSTRUCTING CRYSTAL BASIS VECTORS
    X1C[0] = xc
    X1C[1] = yc
    X1C[2] = zc
    th = acos(X1C[2])
    ph = atan2(X1C[1], X1C[0])
    sth = sin(th)
    cth = cos(th)
    sph = sin(ph)
    cph = cos(ph)
    aux1[0] = sph
    aux1[1] = -cph
    aux1[2] = 0
    aux2[0] = cph * cth
    aux2[1] = sph * cth
    aux2[2] = -sth
    lm = np.random.uniform(0, 2 * pi, 1)
    clm = cos(lm)
    slm = sin(lm)
    X2C = clm * aux1 + slm * aux2
    X3C = np.cross(X1C, X2C) 

    # CONSTRUCTING SAMPLE BASIS VECTORS
    th = acos(zs)
    ph = atan2(ys, xs)
    sth = sin(th)
    cth = cos(th)
    sph = sin(ph)
    cph = cos(ph)
    Aux1[0] = sph
    Aux1[1] = -cph
    Aux1[2] = 0
    Aux2[0] = cph * cth
    Aux2[1] = sph * cth
    Aux2[2] = -sth
    
    STH = sin(thetamax)
    CTH = cos(thetamax)
    
    PHI = np.random.uniform(0, 2 * pi, p)
    Lambdaa = np.random.uniform(0, 2 * pi, p)
    Theta = met.INVCDF1D(axisdist, thetamin, thetamax, p)

   
    for k in range(p):
        ph = PHI[k]
        th = Theta[k]
        SPH = sin(ph)
        CPH = cos(ph)
        STH = sin(th)
        CTH = cos(th)
        sthcph = STH * CPH
        sthsph = STH * SPH
        X1S[0] = sthcph * Aux1[0] + sthsph * Aux2[0] + CTH * xs
        X1S[1] = sthcph * Aux1[1] + sthsph * Aux2[1] + CTH * ys
        X1S[2] = sthcph * Aux1[2] + sthsph * Aux2[2] + CTH * zs
        th = acos(X1S[2])
        ph = atan2(X1S[1], X1S[0])
        sth = sin(th)
        cth = cos(th)
        sph = sin(ph)
        cph = cos(ph)
        aux1[0] = sph
        aux1[1] = -cph
        aux1[2] = 0
        aux2[0] = cph * cth
        aux2[1] = sph * cth
        aux2[2] = -sth
        
        lm = Lambdaa[k]
        clm = cos(lm)
        slm = sin(lm)
        X2S[0] = clm * aux1[0] + slm * aux2[0]
        X2S[1] = clm * aux1[1] + slm * aux2[1]
        X2S[2] = clm * aux1[2] + slm * aux2[2]
           
        X3S[0] = X1S[1] * X2S[2] - X2S[1] * X1S[2]
        X3S[1] = X1S[2] * X2S[0] - X2S[2] * X1S[0]
        X3S[2] = X1S[0] * X2S[1] - X2S[0] * X1S[1]
        
        
        den1 = X2S[1] * X3S[2] - X2S[2] * X3S[1]
        den2 = X3S[1] * X1S[2] - X3S[2] * X1S[1]
        den3 = X1S[1] * X2S[2] - X1S[2] * X2S[1]
        den = sqrt (den1 ** 2 + den2 ** 2 + den3 ** 2)
                
        alpha = den1 / den
        beta = den2 / den
        gamma = den3 / den
        
        
        
        Rot[k, 0, 0] = alpha * X1C[0] + beta * X2C[0] + gamma * X3C[0]
        Rot[k, 1, 0] = alpha * X1C[1] + beta * X2C[1] + gamma * X3C[1]
        Rot[k, 2, 0] = alpha * X1C[2] + beta * X2C[2] + gamma * X3C[2]
        
        
        
        den1 = X2S[2] * X3S[0] - X2S[0] * X3S[2]
        den2 = X3S[2] * X1S[0] - X3S[0] * X1S[2]
        den3 = X1S[2] * X2S[0] - X1S[0] * X2S[2]
        den = sqrt (den1 ** 2 + den2 ** 2 + den3 ** 2)
        
        alpha = den1 / den
        beta = den2 / den
        gamma = den3 / den
        
        Rot[k, 0, 1] = alpha * X1C[0] + beta * X2C[0] + gamma * X3C[0]
        Rot[k, 1, 1] = alpha * X1C[1] + beta * X2C[1] + gamma * X3C[1]
        Rot[k, 2, 1] = alpha * X1C[2] + beta * X2C[2] + gamma * X3C[2]
        
        
        
        den1 = X2S[0] * X3S[1] - X2S[1] * X3S[0]
        den2 = X3S[0] * X1S[1] - X3S[1] * X1S[0]
        den3 = X1S[0] * X2S[1] - X1S[1] * X2S[0]
        den = sqrt (den1 ** 2 + den2 ** 2 + den3 ** 2)
        
        
        alpha = den1 / den
        beta = den2 / den
        gamma = den3 / den
        
        Rot[k, 0, 2] = alpha * X1C[0] + beta * X2C[0] + gamma * X3C[0]
        Rot[k, 1, 2] = alpha * X1C[1] + beta * X2C[1] + gamma * X3C[1]
        Rot[k, 2, 2] = alpha * X1C[2] + beta * X2C[2] + gamma * X3C[2]
        
    
    Eul = conv.rm_to_zxz(Rot)
    if makecanvas == 0:
        
        obj.Eul = Eul
    else:
        obj.EUL = getcanvas(m, n, Eul)
            
    toc = time.time()
    obj.exetime = toc - tic 
    return obj
  
def Texture_Sheet(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int makecanvas = obj.makecanvas
    cdef double tic, toc
    
    cdef float eul1 = obj.eul1
    cdef float eul = obj.eul
    cdef float eul2 = obj.eul2
    cdef float gammamin = obj.gammamin
    cdef float gammamax = obj.gammamax
    cdef float thetamin = obj.thetamin
    cdef float thetamax = obj.thetamax
    cdef float phimin = obj.phimin
    cdef float phimax = obj.phimax
    gammadist = obj.gammadist
    thetadist = obj.thetadist
    phidist = obj.phidist
    
    
    cdef np.ndarray[np.float64_t, ndim = 2] Axis = np.zeros((p, 3), dtype=np.float64)

    plantseed(myseed)
    tic = time.time()

    Angle = met.INVCDF1D(gammadist, gammamin, gammamax, p)
    theta = met.INVCDF1D(thetadist, thetamin, thetamax, p)
    phi = met.INVCDF1D(phidist, phimin, phimax, p)
    Axis[:, 0] = np.sin(theta) * np.cos(phi)
    Axis[:, 1] = np.sin(theta) * np.sin(phi)
    Axis[:, 2] = np.cos(theta)
    Roo = conv.zxz_to_rm(np.array([[eul1], [eul], [eul2]]).T)
    Ro = Roo[0, :, :]
    delR = conv.aap_to_rm(Axis, Angle)
    
    Rot = np.einsum('kil,lj->kij', delR , Ro)
        
    Eul = conv.rm_to_zxz(Rot)

    if makecanvas == 0:
        
        obj.Eul = Eul
    else:
        obj.EUL = getcanvas(m, n, Eul)
            
    toc = time.time()
    obj.exetime = toc - tic 
    return obj


def Texture_Generic_Euler(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int makecanvas = obj.makecanvas
    cdef double tic, toc
    
    cdef int grainloc = obj.grainloc
    Phi1Func = obj.Phi1Func
    PhiFunc = obj.PhiFunc
    Phi2Func = obj.Phi2Func
      
    
    
    cdef int i
    cdef float th, ph, lm , sth, sph, cth, cph, clm, slm, Xc0, Xc1, Xc2
    cdef np.ndarray[np.float64_t, ndim = 1] lambdaa = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] phi = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] V1 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] V2 = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Xc = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Yc = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Zc = np.zeros(3, dtype=np.float64)
    

    # DESCRIPTORS OF ORIENTATIONS  
    cdef np.ndarray[np.float64_t, ndim = 3] Rot = np.zeros((p, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] delR = np.zeros((3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Eul = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Axis = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Angle = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Nd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rd = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Rv = np.zeros((p, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] Q = np.zeros((p, 4), dtype=np.float64)
    
    plantseed(myseed)
    tic = time.time()
    


cdef np.ndarray[np.float64_t, ndim = 1] ProbSinSquare(float low, float high, float m, int p):
    
    
    cdef int i, j
    cdef float g, u, v1, v2, g1, g2, mbt, N, MIN
    cdef int num = 1000
    cdef np.ndarray[np.float64_t, ndim = 1] GRID = np.zeros(num, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] VALUE = np.zeros(num, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] U = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] X = np.zeros(p, dtype=np.float64)
   
    U = np.random.uniform(0, 1, p)
    
    GRID = np.linspace(low, high, num)
    mbt = m / 2
    N = ((high - mbt * sin(high / mbt)) - (low - mbt * sin(low / mbt)))
    MIN = low - mbt * sin(low / mbt)
    for i in range(num):
        g = GRID[i]
        VALUE[i] = (g - mbt * sin(g / mbt) - MIN) / N


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


cdef getcanvas(int m, int n, np.ndarray[np.float64_t, ndim=2] Eul):
    cdef int i, j, count    
    cdef np.ndarray[np.float64_t, ndim = 3] EUL = np.zeros((m, n, 3), dtype=np.float64)
    count = 0
    for i in range(m):
        for j in range(n):
           
            EUL[i, j, 0 ] = Eul[count, 0]
            EUL[i, j, 1 ] = Eul[count, 1]
            EUL[i, j, 2 ] = Eul[count, 2]
            count = count + 1
    return EUL
