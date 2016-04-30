from __future__ import division

import collections
import math
import os
import platform
import signal
import sys
import time
import sip
os.environ['QT_API'] = 'pyqt'
sip.setapi('QDateTime', 2)
sip.setapi("QDate", 2)
sip.setapi("QString", 2)
sip.setapi("QVariant", 2)
sip.setapi('QTextStream', 2)
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'


import cv2
from matplotlib.mlab import griddata
from matplotlib.patches import Rectangle
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
import sip
from skimage import feature
from skimage.filters import sobel
from skimage.measure import label, regionprops, find_contours
from sympy import lambdify

import libASSIGN
import libCOLOR
import libCONVERT
import libDUMMY
import libEVOLVE
import libMETROPOLIS
import libMORPH
import libTEXT
import libTEXTURIZE
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mayavi.mlab as maya
import numpy as np
import scipy as sc
import sympy as sp


pi = np.pi  