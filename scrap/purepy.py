from math import *
import time

import numpy as np
import scipy as sc


I1 = np.random.random((3, 3))
I2 = np.random.random((3, 3))
Ts = np.random.random((24, 3, 3))
TT = np.random.random((576, 3, 3))
Tst = np.random.random((576, 3, 3))
I1 = np.array(I1, dtype=np.float64)
I2 = np.array(I2, dtype=np.float64)
I1t = np.zeros((24, 3, 3))
I2t = np.zeros((24, 3, 3))
I1v = np.zeros((24, 3, 3))
I2v = np.zeros((24, 3, 3))
I1t = np.array(I1t, dtype=np.float64)
I2t = np.array(I2t, dtype=np.float64)
I1v = np.array(I1v, dtype=np.float64)
I2v = np.array(I2v, dtype=np.float64)
TT = np.array(TT, dtype=np.float64)
Tst = np.array(Tst, dtype=np.float64)
Ts = np.array(Ts, dtype=np.float64)
theta = np.zeros((576,))
delR = np.zeros((576, 3, 3))
T = np.zeros((576,))

p = 100
tt = p * (p + 1) / 2
# tt = 5000

t1 = time.time()
Tst = np.transpose(Ts, (0, 2, 1))
count = 0

I1t = np.transpose(I1)
for i in xrange(24):
    for j in xrange(24):
        TT[count, :, :] = np.dot(Tst[i, :, :], Ts[j, :, :])
        count = count + 1
t2 = time.time()

for k in xrange(tt):
    R1 = np.einsum('ij,kji->kij', I1t, TT)
    delR = np.einsum('ijk,kj->ijk', R1, I2)

for k in xrange(576):
    theta[k] = np.arccos((np.trace(delR[k, :, :] - 1) / 2))
print (theta)    
# R1 = np.einsum('ijk,kj->ijk', TT, I1t)
# R1 = np.tensordot(I1t, TT, axes=[1, 1]).swapaxes(0, 1)  # (, , I2)
# delR = np.tensordot(R1, I2, axes=[1, 1]).swapaxes(0, 1)                 
#


# print(delR.shape)    
t3 = time.time()
print(t2 - t1)
print(t3 - t2)
