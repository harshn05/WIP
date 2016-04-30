# import numpy as np
# import time
# from math import *
# import libTEXTURIZE as lt
# import libANATEXT as at 
# import matplotlib.pyplot as plt
# import texture as txt
# # from libTEXTURIZE import *
# 
# class Mock():
#     pass
#  
# 
# Tcube = at.TmatricesCubic()
# symnum = 24
# symcomb = symnum ** 2
# TT = np.zeros((symcomb, 3, 3))
# 
# p =200
# obj = Mock()
# obj.m = 1
# obj.n = 1
# obj.p = p
# obj.myseed = 1
# 
# 
# times = p * (p + 1) / 2
# theta = np.zeros(times)
# Rot = txt.uro_xyz(p, 0)
#     
# lt.uro_xyz(obj)
# #Rot = obj.Rot
# tstart = time.time()
# theta = np.zeros(times, dtype=np.float64)
# THETA = np.zeros(times, dtype=np.float64)
# delR = np.zeros((times, 3, 3), dtype=np.float64)
# mycos = np.zeros(symcomb, dtype=np.float64)
# Tcubet = np.transpose(Tcube, (0, 2, 1))
# count = 0
# for i in xrange(symnum):
#     for j in xrange(symnum):
#         TT[count, :, :] = np.dot(Tcubet[i, :, :], Tcube[j, :, :])
#         count = count + 1
# 
# count = 0
# I = np.eye(3)
# Rott = np.transpose(Rot, (0, 2, 1))
# for ii in range(p):
#     for jj in range(ii + 1, p):
#         delR = np.einsum('kil,lj->kij', np.einsum('il,klj->kij', Rott[jj, :, :], TT), Rot[ii, :, :])
#         mycos = (np.einsum('kij,ji->k', delR, I) - 1) / 2
#         A = np.max(mycos)
#         theta[count] = acos(A)
#         count = count + 1
# tend = time.time()
# 
# theta = theta * 180 / np.pi
# 
theta = np.ma.masked_equal(theta, 0)
print(tend - tstart)
plt.hist(theta.compressed(), bins=50, normed=1, color='green', alpha=0.8)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.xlabel(r"$\theta$",fontsize=16)
# plt.ylabel(r"$P(\theta)$",fontsize=16)
plt.grid()
plt.show()

