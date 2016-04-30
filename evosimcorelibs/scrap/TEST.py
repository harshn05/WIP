from mpl_toolkits.mplot3d import Axes3D
from sympy import lambdify
from sympy.abc import *

import libsoMH as mh
import matplotlib.pyplot as plt
import numpy as np 


'''3D METROPOLIS'''
'''
f=lambdify('x,y,z','x**6','numpy')
X,Y,Z=mh.MH3D(f,0,1,0,1,0,1,10000)
#print X.shape
#plt.scatter(X,Y,'.')
#plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

'''
'''2D METROPOLIS'''
'''
f=lambdify('x,y','x**6','numpy')
X,Y=mh.MH2D(f,0,1,0,1,1000)
fig = plt.figure()
plt.scatter(X,Y)
plt.show()
'''
'''1D METROPOLIS'''
'''
f=lambdify('x','x**6','numpy')
X=mh.MH1D(f,0,1,1000)
fig = plt.figure()
plt.plot(X,'.')
plt.show()
'''
