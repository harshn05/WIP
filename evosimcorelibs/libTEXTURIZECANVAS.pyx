# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
from __future__ import division

from cmath import sqrt
from math import gamma
import math
import os
import random
import sys

import cv2

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




def plantseed(myseed):
    if myseed > 0:
        np.random.seed(myseed)
        random.seed(myseed)

def uro_xyz(obj):
    cdef int myseed = obj.myseed
    cdef int m = obj.m
    cdef int n = obj.n
    cdef int p = m * n
    
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
    Axis, Angle = conv.rm_to_aap(Rot)
    Nd, Rd = conv.rm_to_io(Rot)
    Rv = conv.aap_to_rv(Axis, Angle)
    Q = conv.aap_to_qn(Axis, Angle)
 

    obj.Rot = Rot
    obj.Eul = Eul
    obj.Axis = Axis
    obj.Angle = Angle
    obj.Nd = Nd
    obj.Rd = Rd
    obj.Rv = Rv
    obj.Q = Q
    return obj
        
cdef makecanvas(int m, int n, np.ndarray[np.float64_t, ndim=3] Rot, np.ndarray[np.float64_t, ndim=2] Eul, np.ndarray[np.float64_t, ndim=2] Axis, np.ndarray[np.float64_t, ndim=1] Angle, np.ndarray[np.float64_t, ndim=2] Nd, np.ndarray[np.float64_t, ndim=2] Rd, np.ndarray[np.float64_t, ndim=2] Rv, np.ndarray[np.float64_t, ndim=2] Q):
    
    cdef int i, j, count
    
    cdef np.ndarray[np.float64_t, ndim = 4] ROT = np.zeros((m, n, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] EUL = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] AXIS = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] ANGLE = np.zeros(m, n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] ND = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] RD = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] RV = np.zeros((m, n, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] QQ = np.zeros((m, n, 4), dtype=np.float64)
    count = 0
    for i in range(m):
        for j in range(n):
            ROT[i, j, 0, 0] = Rot[count, 0, 0]
            ROT[i, j, 0, 1] = Rot[count, 0, 1]
            ROT[i, j, 0, 2] = Rot[count, 0, 2]
            ROT[i, j, 1, 0] = Rot[count, 1, 0]
            ROT[i, j, 1, 1] = Rot[count, 1, 1]
            ROT[i, j, 1, 2] = Rot[count, 1, 2]
            ROT[i, j, 2, 0] = Rot[count, 2, 0]
            ROT[i, j, 2, 1] = Rot[count, 2, 1]
            ROT[i, j, 2, 2] = Rot[count, 2, 2]
            EUL[i, j, 0 ] = Eul[count, 0]
            EUL[i, j, 1 ] = Eul[count, 1]
            EUL[i, j, 2 ] = Eul[count, 2]
            AXIS[i, j, 0 ] = Axis[count, 0]
            AXIS[i, j, 1 ] = Axis[count, 1]
            AXIS[i, j, 2 ] = Axis[count, 2]
            ANGLE[i, j] = Angle[count]
            ND[i, j, 0 ] = Nd[count, 0]
            ND[i, j, 1 ] = Nd[count, 1]
            ND[i, j, 2 ] = Nd[count, 2]
            RD[i, j, 0 ] = Nd[count, 0]
            RD[i, j, 1 ] = Rd[count, 1]
            RD[i, j, 2 ] = Rd[count, 2]
            RV[i, j, 0 ] = Rv[count, 0]
            RV[i, j, 1 ] = Rv[count, 1]
            RV[i, j, 2 ] = Rv[count, 2]
            QQ[i, j, 0 ] = Q[count, 0]
            QQ[i, j, 1 ] = Q[count, 1]
            QQ[i, j, 2 ] = Q[count, 2]
            QQ[i, j, 3 ] = Q[count, 3]
            count = count + 1
    return ROT, EUL, AXIS, ANGLE, ND, RD, RV, QQ
             
 
def uro_nxyz(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    
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
    Axis, Angle = conv.rm_to_aap(Rot)
    Nd, Rd = conv.rm_to_io(Rot)
    Rv = conv.aap_to_rv(Axis, Angle)
    Q = conv.aap_to_qn(Axis, Angle)
    

    obj.Rot = Rot
    obj.Eul = Eul
    obj.Axis = Axis
    obj.Angle = Angle
    obj.Nd = Nd
    obj.Rd = Rd
    obj.Rv = Rv
    obj.Q = Q
    return obj
        
        
def uro_arvo(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    
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
    Axis, Angle = conv.rm_to_aap(Rot)
    Nd, Rd = conv.rm_to_io(Rot)
    Rv = conv.aap_to_rv(Axis, Angle)
    Q = conv.aap_to_qn(Axis, Angle)
    

    obj.Rot = Rot
    obj.Eul = Eul
    obj.Axis = Axis
    obj.Angle = Angle
    obj.Nd = Nd
    obj.Rd = Rd
    obj.Rv = Rv
    obj.Q = Q
    return obj


def uro_mackenzie(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    
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
    Axis, Angle = conv.rm_to_aap(Rot)
    Nd, Rd = conv.rm_to_io(Rot)
    Rv = conv.aap_to_rv(Axis, Angle)
    Q = conv.aap_to_qn(Axis, Angle)
    

    obj.Rot = Rot
    obj.Eul = Eul
    obj.Axis = Axis
    obj.Angle = Angle
    obj.Nd = Nd
    obj.Rd = Rd
    obj.Rv = Rv
    obj.Q = Q
    return obj


def uro_shoemake1(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    
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
    Nd, Rd = conv.rm_to_io(Rot)
    Rv = conv.aap_to_rv(Axis, Angle)
    Q = conv.aap_to_qn(Axis, Angle)
    

    obj.Rot = Rot
    obj.Eul = Eul
    obj.Axis = Axis
    obj.Angle = Angle
    obj.Nd = Nd
    obj.Rd = Rd
    obj.Rv = Rv
    obj.Q = Q
    return obj


def uro_shoemake2(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    
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
    Nd, Rd = conv.rm_to_io(Rot)
    Rv = conv.aap_to_rv(Axis, Angle)
    Q = conv.aap_to_qn(Axis, Angle)
    

    obj.Rot = Rot
    obj.Eul = Eul
    obj.Axis = Axis
    obj.Angle = Angle
    obj.Nd = Nd
    obj.Rd = Rd
    obj.Rv = Rv
    obj.Q = Q
    return obj

def uro_axisangle(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    
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
    Nd, Rd = conv.rm_to_io(Rot)
    Rv = conv.aap_to_rv(Axis, Angle)
    Q = conv.aap_to_qn(Axis, Angle)
    

    obj.Rot = Rot
    obj.Eul = Eul
    obj.Axis = Axis
    obj.Angle = Angle
    obj.Nd = Nd
    obj.Rd = Rd
    obj.Rv = Rv
    obj.Q = Q
    return obj


def uro_euler(obj):
    cdef int myseed = obj.myseed
    cdef int p = obj.p
    cdef int m = obj.m
    cdef int n = obj.n
    
    
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

    Eul[0:, 0] = np.random.uniform(0, 2 * pi, [p, ])
    r1 = np.random.uniform(0, 1, [p, ])
    Eul[0:, 1] = np.arccos(2 * r1 - 1)
    Eul[0:, 2] = np.random.uniform(0, 2 * pi, [p, ])
   
    Rot = conv.zxz_to_rm(Eul)
    Axis, Angle = conv.rm_to_aap(Rot)
    Nd, Rd = conv.rm_to_io(Rot)
    Rv = conv.aap_to_rv(Axis, Angle)
    Q = conv.aap_to_qn(Axis, Angle)
    

    obj.Rot = Rot
    obj.Eul = Eul
    obj.Axis = Axis
    obj.Angle = Angle
    obj.Nd = Nd
    obj.Rd = Rd
    obj.Rv = Rv
    obj.Q = Q
    return obj

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
