from sympy import lambdify
from sympy.abc import *

import libsoMH as mh
import numpy as np


myfunc1 = lambdify('x', x, modules=['numpy', 'sympy'])
myfunc2 = lambdify('x,y', x * y, modules=['numpy', 'sympy'])
myfunc3 = lambdify('x,y,z', x * y * z, modules=['numpy', 'sympy'])
# X =mh.MH1D(myfunc1,0,1000,100)
