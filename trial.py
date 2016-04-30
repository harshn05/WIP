import numpy, random

from mayavi.mlab import *


def cMap(x,y,z):
    #whatever logic you want for colors
    return [random.random() for i in x]

def test_points3d():
    t = numpy.linspace(0, 4*numpy.pi, 20)
    cos = numpy.cos
    sin = numpy.sin

    x = sin(2*t)
    y = cos(t)
    z = cos(2*t)
    s = cMap(x,y,z)

    return points3d(x, y, z, s, colormap="spectral", scale_factor=0.25)

test_points3d()
