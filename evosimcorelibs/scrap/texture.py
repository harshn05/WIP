# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
from __future__ import division

import math
import os
import random
import sys
import time

import libCONVERT as conv
import numpy as np


pibt = np.pi / 2
tpibth = 2 * np.pi / 3
fpibth = 4 * np.pi / 3
pi = np.pi
thpibt = 3 * np.pi / 2
tpi = 2 * np.pi



def plantseed(myseed):
    if myseed > 0:
        np.random.seed(myseed)
        random.seed(myseed)

def uro_xyz(p, myseed):
    
    V1 = np.zeros((p, 3))
    V2 = np.zeros((p, 3))
    Xc = np.zeros((p, 3))
    Yc = np.zeros((p, 3))
    Rot = np.zeros((p, 3, 3))
    
    
    plantseed(myseed)
    t1 = time.time()
    r1 = np.random.uniform(0, 1, [p, ])
    theta = np.arccos(2 * r1 - 1)
    phi = np.random.uniform(0, 2 * np.pi, [p, ])
    lambdaa = np.random.uniform(0, 2 * pi, [p, ])
    CL = np.cos(lambdaa)
    SL = np.sin(lambdaa)
    
    ST = np.sin(theta)
    CT = np.cos(theta)
    SP = np.sin(phi)
    CP = np.cos(phi)
    
    Xc[:, 0] = ST * CP
    Xc[:, 1] = ST * SP
    Xc[:, 2] = CT
    
    V1[:, 0] = SP
    V1[:, 1] = -CP
    V1[:, 2] = 0
    
    V2[:, 0] = CP * CT
    V2[:, 1] = SP * CT
    V2[:, 2] = -ST

    Yc[:, 0] = CL * V1[:, 0] + SL * V2[:, 0]
    Yc[:, 1] = CL * V1[:, 1] + SL * V2[:, 1]
    Yc[:, 2] = CL * V1[:, 2] + SL * V2[:, 2]
    Zc = np.cross(Xc, Yc)
    
    Rot[:, 0, :] = Xc[:, :] 
    Rot[:, 1, :] = Yc[:, :]
    Rot[:, 2, :] = Zc[:, :]
    
    Eul = conv.rm_to_zxz(Rot)
    Axis, Angle = conv.rm_to_aap(Rot)
    Nd, Rd = conv.rm_to_io(Rot)
    Rv = conv.aap_to_rv(Axis, Angle)
    Q = conv.aap_to_qn(Axis, Angle)
 
    t2 = time.time()
    print(t2 - t1)
    return Rot
        

Rot = uro_xyz(1000000, 0)
# plt.hist(Eul[:,2])
# plt.show()
