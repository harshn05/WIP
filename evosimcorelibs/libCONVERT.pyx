# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True
from __future__ import division

import os
import sys

import numpy as np


from libc.math cimport sin, cos, tan, acos, atan, abs, sqrt
cimport numpy as np
cimport cython

cdef float pi = np.pi
cdef float tpi = 2 * pi
cdef float pibt = pi / 2
cdef float thpibt = 3 * pi / 2


def zxz_to_rm(np.ndarray[np.float64_t, ndim=2] PHI):
    cdef float c1, c, c2, s1, s, s2, c1c2, s1s2, s1c2, s2c1
    cdef int l, k
    l = PHI.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 3] R = np.zeros((l, 3, 3), dtype=np.float64)

    for k in range(l):
        c1 = cos(PHI[k, 0])
        c = cos(PHI[k, 1])
        c2 = cos(PHI[k, 2])
        s1 = sin(PHI[k, 0])
        s = sin(PHI[k, 1])
        s2 = sin(PHI[k, 2])
        c1c2 = c1 * c2
        s1s2 = s1 * s2
        s1c2 = s1 * c2
        s2c1 = s2 * c1
        R[k, 0, 0] = c1c2 - s1s2 * c
        R[k, 0, 1] = s1c2 + s2c1 * c
        R[k, 0, 2] = s2 * s
        R[k, 1, 0] = -s2c1 - s1c2 * c
        R[k, 1, 1] = -s1s2 + c1c2 * c
        R[k, 1, 2] = c2 * s
        R[k, 2, 0] = s * s1
        R[k, 2, 1] = -s * c1
        R[k, 2, 2] = c
    return R


def rm_to_aap(np.ndarray[np.float64_t, ndim=3] R):

    cdef int l, k
    l = R.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 1] angle = np.zeros(l, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] mytrace = np.zeros(l, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Mag = np.zeros(l, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] axis = np.zeros((l, 3), dtype=np.float64)
    mytrace = (np.einsum('kii', R) - 1) / 2
    angle = np.arccos(mytrace)
    for k in range(l):
        axis[k, 0] = (R[k, 1, 2] - R[k, 2, 1]) 
        axis[k, 1] = (R[k, 2, 0] - R[k, 0, 2]) 
        axis[k, 2] = (R[k, 0, 1] - R[k, 1, 0]) 
        
    Mag = np.sqrt((axis * axis).sum(axis=1))

    for k in range(l):
        axis[k, :] = axis[k, :] / Mag[k] 
    
    return axis, angle


def rm_to_zxz(np.ndarray[np.float64_t, ndim=3] R):
    cdef int k, l
    cdef float r31, r32, r23, r13, a
    l = R.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 2] PHI = np.zeros((l, 3), dtype=np.float64)
    PHI = np.zeros((l, 3))

    for k in range(l):
        r31 = R[k, 2, 0]
        r32 = R[k, 2, 1]
        r23 = R[k, 1, 2]
        r13 = R[k, 0, 2]

        PHI[k, 1] = acos(R[k, 2, 2])

        a = abs(atan(r31 / r32))

        if r32 <= 0 and r31 >= 0:
            PHI[k, 0] = a
        elif r32 >= 0 and r31 >= 0:
            PHI[k, 0] = pi - a
        elif r32 >= 0 and r31 <= 0:
            PHI[k, 0] = pi + a
        elif r32 <= 0 and r31 <= 0:
            PHI[k, 0] = tpi - a

        a = abs(atan(r13 / r23))

        if r23 >= 0 and r13 >= 0:
            PHI[k, 2] = a
        elif r23 <= 0 and r13 >= 0:
            PHI[k, 2] = pi - a
        elif r23 <= 0 and r13 <= 0:
            PHI[k, 2] = pi + a
        elif r23 >= 0 and r13 <= 0:
            PHI[k, 2] = tpi - a
    return PHI


def zxz_to_aap(np.ndarray[np.float64_t, ndim=2] PHI):
    R = zxz_to_rm(PHI)
    axis, angle = rm_to_aap(R)
    return axis, angle


def aap_to_rm(np.ndarray[np.float64_t, ndim=2] axis, np.ndarray[np.float64_t, ndim=1]  angle):
    cdef int k, l
    cdef float s, c, e1, e2, e3, mag
    l = angle.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 3] R = np.zeros((l, 3, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Mag = np.zeros(l, dtype=np.float64)
    R = np.zeros((l, 3, 3))
    Mag = np.sqrt((axis * axis).sum(axis=1))
    for k in range(l):
        mag = Mag[k]
        e1 = axis[k, 0] / mag
        e2 = axis[k, 1] / mag
        e3 = axis[k, 2] / mag
        s = sin(angle[k])
        c = cos(angle[k])
        R[k, 0, 0] = (1 - c) * e1 * e1 + c
        R[k, 0, 1] = (1 - c) * e1 * e2 + e3 * s
        R[k, 0, 2] = (1 - c) * e1 * e3 - e2 * s
        R[k, 1, 0] = (1 - c) * e2 * e1 - e3 * s
        R[k, 1, 1] = (1 - c) * e2 * e2 + c
        R[k, 1, 2] = (1 - c) * e2 * e3 + e1 * s
        R[k, 2, 0] = (1 - c) * e3 * e1 + e2 * s
        R[k, 2, 1] = (1 - c) * e3 * e2 - e1 * s
        R[k, 2, 2] = (1 - c) * e3 * e3 + c
    return R


def aap_to_zxz(np.ndarray[np.float64_t, ndim=2] axis, np.ndarray[np.float64_t, ndim=1] angle):
    R = aap_to_rm(axis, angle)
    PHI = rm_to_zxz(R)
    return PHI


def aap_to_rv(np.ndarray[np.float64_t, ndim=2] axis, np.ndarray[np.float64_t, ndim=1] angle):
    cdef int l, k
    cdef float coeff
    l = angle.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 2] r = np.zeros((l, 3), dtype=np.float64)
    for k in range(l):
        coeff = tan(angle[k] / 2)
        r[k, 0] = axis[k, 0] * coeff
        r[k, 1] = axis[k, 1] * coeff
        r[k, 2] = axis[k, 2] * coeff
    return r


def rv_to_aap(np.ndarray[np.float64_t, ndim=2] r):
    cdef int l, k
    cdef float mytan
    l = r.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 2] axis = np.zeros((l, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] angle = np.zeros(l, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Mag = np.zeros(l, dtype=np.float64)
    axis = np.zeros((l, 3))
    angle = np.zeros(l)
    Mag = np.sqrt((r * r).sum(axis=1))
    for k in range(l):
        mytan = Mag[k]
        axis[k, 0] = r[k, 0] / mytan
        axis[k, 1] = r[k, 1] / mytan
        axis[k, 2] = r[k, 2] / mytan
        angle[k] = atan(mytan)
    return axis, angle


def rv_to_zxz(np.ndarray[np.float64_t, ndim=2] r):
    axis, angle = rv_to_aap(r)
    PHI = aap_to_zxz(axis, angle)
    return PHI


def rv_to_rm(np.ndarray[np.float64_t, ndim=2] r):
    axis, angle = rv_to_aap(r)
    R = aap_to_rm(axis, angle)
    return R


def zxz_to_rv(np.ndarray[np.float64_t, ndim=2] PHI):
    axis, angle = zxz_to_aap(PHI)
    r = aap_to_rv(axis, angle)
    return r


def rm_to_rv(np.ndarray[np.float64_t, ndim=3] R):
    axis, angle = rm_to_aap(R)
    r = aap_to_rv(axis, angle)
    return r


def io_to_rm(np.ndarray[np.float64_t, ndim=2] nd, np.ndarray[np.float64_t, ndim=2] rd):
    cdef int k, l
    cdef float m1, m2, n0, n1, n2, r0, r1, r2, t0, t1, t2
    l = nd.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 1] magnd = np.zeros(l, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] magrd = np.zeros(l, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 3] R = np.zeros((l, 3, 3), dtype=np.float64)
    magnd = np.sqrt((nd * nd).sum(axis=1))
    magrd = np.sqrt((rd * rd).sum(axis=1))

    for k in range(l):
        m1 = magnd[k]
        m2 = magrd[k]
        n0 = nd[k, 0] / m1
        n1 = nd[k, 1] / m1
        n2 = nd[k, 2] / m1
        r0 = rd[k, 0] / m2
        r1 = rd[k, 1] / m2
        r2 = rd[k, 2] / m2
        t0 = n1 * r2 - n2 * r1
        t1 = n2 * r0 - n0 * r2
        t2 = n0 * r1 - n1 * r0

        R[k, 0, 0] = r0
        R[k, 1, 0] = r1
        R[k, 2, 0] = r2
        R[k, 0, 1] = t0
        R[k, 1, 1] = t1
        R[k, 2, 1] = t2
        R[k, 0, 2] = n0
        R[k, 1, 2] = n1
        R[k, 2, 2] = n2
    return R


def rm_to_io(np.ndarray[np.float64_t, ndim=3] R):
    cdef int l, k
    l = R.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 2] nd = np.zeros((l, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 2] rd = np.zeros((l, 3), dtype=np.float64)

    for k in range(l):
        nd[k, 0] = R[k, 0, 2]
        nd[k, 1] = R[k, 1, 2]
        nd[k, 2] = R[k, 2, 2]
        rd[k, 0] = R[k, 0, 0]
        rd[k, 1] = R[k, 1, 0]
        rd[k, 2] = R[k, 2, 0]
    return nd, rd


def io_to_aap(np.ndarray[np.float64_t, ndim=2] nd, np.ndarray[np.float64_t, ndim=2] rd):
    R = io_to_rm(nd, rd)
    axis, angle = rm_to_aap(R)
    return axis, angle


def io_to_rv(np.ndarray[np.float64_t, ndim=2] nd, np.ndarray[np.float64_t, ndim=2] rd):
    R = io_to_rm(nd, rd)
    r = rm_to_rv(R)
    return r


def io_to_zxz(np.ndarray[np.float64_t, ndim=2] nd, np.ndarray[np.float64_t, ndim=2] rd):
    R = io_to_rm(nd, rd)
    PHI = rm_to_zxz(R)
    return PHI


def aap_to_io(np.ndarray[np.float64_t, ndim=2] axis, np.ndarray[np.float64_t, ndim=1] angle):
    R = aap_to_rm(axis, angle)
    nd, rd = rm_to_io(R)
    return nd, rd


def zxz_to_io(np.ndarray[np.float64_t, ndim=2] PHI):
    R = zxz_to_rm(PHI)
    nd, rd = rm_to_io(R)
    return nd, rd


def rv_to_io(np.ndarray[np.float64_t, ndim=2] r):
    R = rv_to_rm(r)
    nd, rd = rm_to_io(R)
    return nd, rd


def aap_to_qn(np.ndarray[np.float64_t, ndim=2] axis, np.ndarray[np.float64_t, ndim=1] angle):
    cdef int l, k
    cdef float c, s
    l = angle.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 2] q = np.zeros((l, 4), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] Mag = np.zeros(l, dtype=np.float64)
    Mag = np.sqrt((axis * axis).sum(axis=1))

    for k in range(l):
        c = cos(angle[k] / 2)
        s = sin(angle[k] / 2) / Mag[k]
        q[k, 1] = s * axis[k, 0]
        q[k, 2] = s * axis[k, 1]
        q[k, 3] = s * axis[k, 2]
        q[k, 0] = c
    return q


def qn_to_aap(np.ndarray[np.float64_t, ndim=2] q):
    cdef int l, k
    cdef float theta
    l = q.shape[0]
    cdef np.ndarray[np.float64_t, ndim = 2] axis = np.zeros((l, 3), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim = 1] angle = np.zeros(l, dtype=np.float64)

    for k in range(l):
        theta = 2 * acos(q[k, 0])
        s = sin(theta / 2)
        angle[k] = theta
        axis[k, 0] = q[k, 1] / s
        axis[k, 1] = q[k, 2] / s
        axis[k, 2] = q[k, 3] / s
    return axis, angle


def qn_to_zxz(np.ndarray[np.float64_t, ndim=2] q):
    axis, angle = qn_to_aap(q)
    PHI = aap_to_zxz
    return PHI


def qn_to_rm(np.ndarray[np.float64_t, ndim=2] q):
    axis, angle = qn_to_aap(q)
    R = aap_to_rm(axis, angle)
    return R


def qn_to_rv(np.ndarray[np.float64_t, ndim=2] q):
    axis, angle = qn_to_aap(q)
    r = aap_to_rv(axis, angle)
    return r


def qn_to_io(np.ndarray[np.float64_t, ndim=2] q):
    axis, angle = qn_to_aap(q)
    nd, rd = aap_to_io(axis, angle)
    return nd, rd


def zxz_to_qn(np.ndarray[np.float64_t, ndim=2] PHI):
    axis, angle = zxz_to_aap(PHI)
    q = aap_to_qn(axis, angle)
    return q


def rm_to_qn(np.ndarray[np.float64_t, ndim=3] R):
    axis, angle = rm_to_aap(R)
    q = aap_to_qn(axis, angle)
    return q


def rv_to_qn(np.ndarray[np.float64_t, ndim=2] r):
    axis, angle = rv_to_aap(r)
    q = aap_to_qn(axis, angle)
    return q


def io_to_qn(np.ndarray[np.float64_t, ndim=2] nd, np.ndarray[np.float64_t, ndim=2] rd):
    axis, angle = io_to_aap(nd, rd)
    q = aap_to_qn(axis, angle)
    return q
