from __future__ import division

import collections
import os
import platform
import signal
import sys
import time

from IPython.lib import guisupport
from PyQt4 import QtGui, QtCore, uic
from PyQt4.QtCore import pyqtSlot, SIGNAL, SLOT
import cv2
from matplotlib.mlab import griddata
from matplotlib.patches import Rectangle
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_ipython_widget import RichJupyterWidget
from scipy import stats
import sip
from skimage import feature
from skimage.filters import sobel
from skimage.measure import label, regionprops, find_contours
from sklearn import mixture
from sympy import lambdify

from evosimcorelibs import libASSIGN as assign
from evosimcorelibs import libCOLOR as col
from evosimcorelibs import libCONVERT as conv
from evosimcorelibs import libDUMMY as dum
from evosimcorelibs import libEVOLVE as ev
from evosimcorelibs import libMETROPOLIS as met
from evosimcorelibs import libMORPH as am
from evosimcorelibs import libTEXT as at
from evosimcorelibs import libTEXTURIZE as tx
from evosimcorelibs import libTMATRICES as tmat
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import resourceLIST_rc
import scipy.optimize as so
import sympy as sp


# from __builtin__ import str
os.environ['QT_API'] = 'pyqt'
sip.setapi("QString", 2)
sip.setapi("QVariant", 2)

# from PyQt4.QtGui import RichJupyterWidget
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #if needed


#from evosimcorelibs.plotting import latexify

signal.signal(signal.SIGINT, signal.SIG_DFL)


PlottingOptionsUI, PlottingOptionsBase = uic.loadUiType("./uicomponents/PlottingOptions.ui")
pi = np.pi

# DEFAULT_INSTANCE_ARGS = ['qtconsole', '--pylab=inline', '--colors=linux']

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class QIPythonWidget(RichJupyterWidget):
    """ Convenience class for a live IPython console widget. We can replace the standard banner using the customBanner argument"""
    def __init__(self, customBanner=None, *args, **kwargs):
        super(QIPythonWidget, self).__init__()
        if customBanner != None: self.banner = customBanner
        super(QIPythonWidget, self).__init__(*args, **kwargs)
        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.gui = 'qt4'
        self.kernel_client = kernel_client = self._kernel_manager.client()
        kernel_client.start_channels()
        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            guisupport.get_app_qt4().exit()
            
             
        self.exit_requested.connect(stop)

    def pushVariables(self, variableDict):
        """ Given a dictionary containing name / value pairs, push those variables to the IPython console widget """
        self.kernel_manager.kernel.shell.push(variableDict)
    def clearTerminal(self):
        """ Clears the terminal """
        self._control.clear()    
    def printText(self, text):
        """ Prints some plain text to the console """
        self._append_plain_text(text)        
    def executeCommand(self, command):
        """ Execute a command in the frame of the console widget """
        self._execute(command, False)

 
                      

def print_process_id():
    print ('Process ID is:', os.getpid()) 

class EvoSimGui(QtGui.QMainWindow):
        
    def __init__(self, parent=None):
        
        super(EvoSimGui, self).__init__(parent)
        qtCreatorFile = "./uicomponents/EvoSim.ui"  # Enter file here.
        uic.loadUi(qtCreatorFile, self)
        self.show()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.comboBox.setCurrentIndex(1)
        self.sidepanel.setHidden(True)
        
        self.EVOLVE = EvolutionFunctions()
        self.TextureURT = TextureFunctions()
        self.TextureGEN = TextureFunctions()
        self.TextureST = TextureFunctions()
        self.TextureFT = TextureFunctions()
               

        self.msmodel = QtGui.QStandardItemModel(0, 2)
        self.texmodel = QtGui.QStandardItemModel(0, 2)
        self.mswtmodel = QtGui.QStandardItemModel(0, 2)
        self.canvasmodel = QtGui.QStandardItemModel(0, 2)
                
        self.tvMS.setModel(self.msmodel)
        self.tvTEX.setModel(self.texmodel)
        self.tvMSWT.setModel(self.mswtmodel)
        self.tvCANVAS.setModel(self.canvasmodel)
        
        
        
        self.EVOLVEcount = 0
        self.TEXTUREcount = 0
        self.MSWTcount = 0
        self.CANVAScount = 0

        self.tvMS.model().setHorizontalHeaderLabels(['ID', 'Values'])
        self.tvTEX.model().setHorizontalHeaderLabels(['ID', 'Texture'])
        self.tvMSWT.model().setHorizontalHeaderLabels(['ID', 'Texture'])
        self.tvCANVAS.model().setHorizontalHeaderLabels(['ID', 'Texture'])
        
        
       
        self.tvMS.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tvTEX.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tvMSWT.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tvCANVAS.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        
        self.tableTextureBlend.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        self.connect(self.tvMS, QtCore.SIGNAL('customContextMenuRequested(const QPoint&)'), self.CMViewEVOLVE)
        self.connect(self.tvTEX, QtCore.SIGNAL('customContextMenuRequested(const QPoint&)'), self.CMViewTexture)
        self.connect(self.tvMSWT, QtCore.SIGNAL('customContextMenuRequested(const QPoint&)'), self.CMViewMSWT)
        self.connect(self.tvCANVAS, QtCore.SIGNAL('customContextMenuRequested(const QPoint&)'), self.CMViewCANVAS)
        self.connect(self.tableTextureBlend, QtCore.SIGNAL('customContextMenuRequested(const QPoint&)'), self.CMTextureBlend)
        
        
        
        self.textEdit.append("Python Version Info:")
        self.textEdit.append(sys.version)
        self.textEdit.append("")
        self.textEdit.append("Currently Running on: ")
        self.textEdit.append(str(platform.platform()))
        self.textEdit.append("")
        self.textEdit.append("Developed by:")
        self.textEdit.append("Harsh Kumar Narula")
        self.textEdit.append("Indian Institute of Technology Bombay, Mumbai, India")
        self.textEdit.append("Email: harshn@iitb.ac.in")
        self.mydict = [dict(randint=np.random.randint, rand=np.random.uniform, randn=np.random.randn, choice=np.random.choice, max=max, min=min, ceil=np.ceil, floor=np.floor, gamma = np.random.gamma, poission = np.random.poisson, vonmises = np.random.vonmises, gcd = sp.gcd, geq = np.greater_equal), 'math', 'numpy', 'sympy']
        self.texturedict = dict(TEXTURE0='ST', TEXTURE1='FT', TEXTURE2='URT', TEXTURE3='GT')
        self.canvasdict = dict(TEXTURE0='STC', TEXTURE1='FTC', TEXTURE2='URTC', TEXTURE3='GTC')
        
        if self.checkBoxKSM.isChecked():
            self.KSM = 1
        else:
            self.KSM = 0

        self.ipyConsole = QIPythonWidget("Welcome to EvoSim !!!, Powered By:\n\n ")
        #self.layoutipy.addWidget(self.ipyConsole)
        self.ipyConsole.clearTerminal()
        # self.ipyConsole.executeCommand("from mylibs import *")
        # self.ipyConsole.
        self.ipyConsole.executeCommand("from evosimcorelibs.IpythonLibs import *")
        self.PO_colmaps.addItems(plt.colormaps())    
        
        index = self.PO_colmaps.findText("Greys_r", QtCore.Qt.MatchFixedString)
        self.PO_colmaps.setCurrentIndex(index)
                
        # self.mptcolmaps.addItems(mpl.rcParams.keys())
        self.Dumptoipy()
        
    def getpltoptions(self):
        self.colmaps = str(self.PO_colmaps.currentText())
        self.plttitlefontsize = int(self.PO_titlefontsize.text())
        self.pltaxisfontsize = int(self.PO_axisfontsize.text())
        self.plthistbins = int(self.PO_histbins.text())
        self.pltalpha = float(self.PO_alpha.text())
        self.pltmarkersize = int(self.PO_mksize.text())

        
    def getDir(self):
        if self.shouldwesave.isChecked():
            fd = QtGui.QFileDialog.getExistingDirectory()
            if fd == '':
                self.shouldwesave.setChecked(False)
                self.framepath.setEnabled(False)
                self.browsepath.setEnabled(False)

            else:
                self.framepath.setText(fd)

    
    def switchSidePanel(self):
        c = self.tabWidgetMain.currentIndex()
        if c == 0:
            self.sidepanel.setHidden(True)
        else:
            self.sidepanel.setHidden(False)
            
    
    @pyqtSlot(QtCore.QPoint)
    def CMViewEVOLVE(self, point):
        menu = QtGui.QMenu()
        actionGS = menu.addAction("Gray Scale Image")
        actionCOL = menu.addAction("Colored Image")
        actionBOUND = menu.addAction("Show Edges")
        actionTRUE = menu.addAction("Image in OpenCV Coordinate System")
        actionSNS = menu.addAction("Nucleation Sites")
        actionSCEN = menu.addAction("Centroids")
        actionTRIP = menu.addAction("Triple Points")
        actionDISPNC = menu.addAction("Nuclei-Centroid Displacement Field")
        actionDISTNC = menu.addAction("Nuclei-Centroid Distance Distribution")
        actionANGLENC = menu.addAction("Nuclei-Centroid Vector Orientation Distribution")
        actionNND = menu.addAction("Number of Edges Distribution")
        actionLNNCDD = menu.addAction("Fit Normal Distribution to Grain Size")
        actionGAD = menu.addAction("Grain Area Distribution")
        actionDIA = menu.addAction("Grain Diameter Distribution")
        actionPERIDIST = menu.addAction("Perimeter Distribution")
        actionECC = menu.addAction("Eccentricity Distribution")
        actionMORPHORI = menu.addAction("Ellipse Orientation Distribution")
        actionSF = menu.addAction("Shape Factor")
        actionCLEARALL = menu.addAction("Delete All Microstructures")
        actionDELSEL = menu.addAction("Delete Selected Microstructure")
        actionCAW = menu.addAction("Close All Windows")
        
        self.connect(actionDELSEL, SIGNAL("triggered()"), self, SLOT("DeleteThisMS()"))
        self.connect(actionCLEARALL, SIGNAL("triggered()"), self, SLOT("ClearAllMS()"))
        self.connect(actionGAD, SIGNAL("triggered()"), self, SLOT("ShowGAD()"))
        self.connect(actionSNS, SIGNAL("triggered()"), self, SLOT("ShowNS()"))
        self.connect(actionSCEN, SIGNAL("triggered()"), self, SLOT("ShowSCEN()"))
        self.connect(actionCAW, SIGNAL("triggered()"), self, SLOT("ShowCAW()"))
        self.connect(actionDISPNC, SIGNAL("triggered()"), self, SLOT("ShowDISPNC()"))
        self.connect(actionDISTNC, SIGNAL("triggered()"), self, SLOT("ShowDISTNC()"))
        self.connect(actionANGLENC, SIGNAL("triggered()"), self, SLOT("ShowANGLENC()"))
        self.connect(actionNND, SIGNAL("triggered()"), self, SLOT("ShowNND()"))
        self.connect(actionLNNCDD, SIGNAL("triggered()"), self, SLOT("ShowLNNCDD()"))
        self.connect(actionPERIDIST, SIGNAL("triggered()"), self, SLOT("ShowPERIDIST()"))
        self.connect(actionECC, SIGNAL("triggered()"), self, SLOT("ShowEccentricityDIST()"))
        self.connect(actionMORPHORI, SIGNAL("triggered()"), self, SLOT("ShowEllOriDIST()"))
        self.connect(actionSF, SIGNAL("triggered()"), self, SLOT("ShowShapeFactor()"))
        self.connect(actionDIA, SIGNAL("triggered()"), self, SLOT("ShowDIA()"))
        self.connect(actionGS, SIGNAL("triggered()"), self, SLOT("ShowGS()"))
        self.connect(actionCOL, SIGNAL("triggered()"), self, SLOT("ShowCOL()"))
        self.connect(actionTRUE, SIGNAL("triggered()"), self, SLOT("ShowOPENCV()"))
        self.connect(actionBOUND, SIGNAL("triggered()"), self, SLOT("ShowBOUND()"))
        self.connect(actionTRIP, SIGNAL("triggered()"), self, SLOT("ShowTRIP()"))
        
        menu.exec_(self.mapToGlobal(point))  
        
    @pyqtSlot(QtCore.QPoint)
    def CMViewMSWT(self, point):
        menu = QtGui.QMenu()
        actionEULER = menu.addAction("Show Euler Space")
        self.connect(actionEULER, SIGNAL("triggered()"), self, SLOT("ShowEulerMSWT()"))
        actionPOLE = menu.addAction("Show Pole Figures")
        self.connect(actionPOLE, SIGNAL("triggered()"), self, SLOT("PoleFigureMSWT()"))
        actionMD = menu.addAction("Show Misorientation Distribution")
        self.connect(actionMD, SIGNAL("triggered()"), self, SLOT("ShowMD()"))
        actionCLEARALL = menu.addAction("Delete All MICROSTRUCTURES")
        self.connect(actionCLEARALL, SIGNAL("triggered()"), self, SLOT("ClearAllMSWT()"))
        actionDELTHIS = menu.addAction("Delete Selected MICROSTRUCTURES")
        self.connect(actionDELTHIS, SIGNAL("triggered()"), self, SLOT("DeleteThisMSWT()"))
        actionCONVERTTOANG = menu.addAction("Convert to *.ang")
        self.connect(actionCONVERTTOANG, SIGNAL("triggered()"), self, SLOT("ANGConvert()"))
        
        
        menu.exec_(self.mapToGlobal(point))
        
    @pyqtSlot()
    def PoleFigureMSWT(self):
        sqt = np.sqrt(2)
        index, mytexture, ms = self.selectedMSWT()
        if index != []:
            TextureURT = getattr(self, mytexture)
            Eul = TextureURT.EULER
            XPx, YPx = self.polefig(Eul, [0, 0, 1])
            XPy, YPy = self.polefig(Eul, [1, 1, 0])
            XPz, YPz = self.polefig(Eul, [1, 1, 1])
            figname = ms + " Pole Figures"
            if plt.fignum_exists(figname):
                plt.close(figname)
            fig = plt.figure(figname)
            axisx = fig.add_subplot(1, 3, 1)
            axisy = fig.add_subplot(1, 3, 2)
            axisz = fig.add_subplot(1, 3, 3)
            axisx.plot(XPx, YPx, '.')
            axisy.plot(XPy, YPy, '.')
            axisz.plot(XPz, YPz, '.')
            
            axisx.set_xlim(-sqt, +sqt)
            axisx.set_ylim(-sqt, +sqt)
            axisy.set_xlim(-sqt, +sqt)
            axisy.set_ylim(-sqt, +sqt)
            axisz.set_xlim(-sqt, +sqt)
            axisz.set_ylim(-sqt, +sqt)
            axisx.set_aspect("equal")
            axisy.set_aspect("equal")
            axisz.set_aspect("equal")
      
            circle = plt.Circle((0, 0), sqt, color='cyan')
            axisx.add_artist(circle)
            circle = plt.Circle((0, 0), sqt, color='cyan')
            axisy.add_artist(circle)
            circle = plt.Circle((0, 0), sqt, color='cyan')
            axisz.add_artist(circle)
            
            axisx.set_title("[0 0 1]", fontsize=25)
            axisy.set_title("[1 1 0]", fontsize=25)
            axisz.set_title("[1 1 1]", fontsize=25)
            axisx.get_xaxis().set_visible(False)
            axisx.get_yaxis().set_visible(False)
            axisy.get_xaxis().set_visible(False)
            axisy.get_yaxis().set_visible(False)
            axisz.get_xaxis().set_visible(False)
            axisz.get_yaxis().set_visible(False)
            
            plt.grid();plt.tight_layout();plt.show()
    
        
    @pyqtSlot()
    def ShowDISPNC(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            X = EVOLVE.X
            Y = EVOLVE.Y
            I = EVOLVE.I
            p = X.shape[0]
            m = I.shape[0]
            n = I.shape[1]
            Xc, Yc = am.centroids(I, m, n, p)
         
            A = np.zeros((p, 4))
            for i in range(p):
                A[i, 0] = X[i]
                A[i, 1] = Y[i]
                A[i, 2] = Xc[i] - X[i]
                A[i, 3] = Yc[i] - Y[i]

            XX, YY, U, V = zip(*A)
            
            plt.figure(ms + " N-C Displacement Field")
            plt.imshow(EVOLVE.Col[:, :, ::-1], origin='upper', interpolation='none')
            plt.quiver(YY, XX, V, U, angles='xy', scale_units='xy', scale=1)  # LITTLE HACK
            plt.xlabel(r"$Y\; axis\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$\leftarrow X\; axis$", fontsize=self.pltaxisfontsize)
            plt.draw()
            plt.grid(True, color='white');plt.tight_layout();plt.show()
            
            
    @pyqtSlot()
    def ShowGS(self):
        cv2.destroyWindow("Microstructure Evolution Random Color")
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            A = getattr(self, myevolve)
            self.getpltoptions()
            plt.figure(ms + " Gray Scale")
            plt.imshow(A.I, interpolation='none', cmap=cm.Greys_r)
            # plt.xlabel(r"$X\; axis\rightarrow$", fontsize=self.pltaxisfontsize)
            # plt.ylabel(r"$Y\; axis\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.tight_layout();plt.show()
            
        
            
    @pyqtSlot()
    def ShowEulerMSWT(self):
        index, mymswt, ms = self.selectedMSWT()
        if index != []:
            mymswt = getattr(self, mymswt)
            I = getattr(mymswt, 'I')
            Eul = getattr(mymswt, 'EULER') * 180 / pi
            fig = plt.figure(ms + "Euler Space")
            ax = fig.gca(projection='3d')
            ax.scatter(Eul[:, 0], Eul[:, 1], Eul[:, 2], c='b', marker='.')
            ax.set_xlim(0, 360)
            ax.set_ylim(0, 180)
            ax.set_zlim(0, 360)                       
            ax.set_aspect('auto', 'datalim')
            ax.set_xlabel(r'$\phi_1  \rightarrow$', fontsize=self.pltaxisfontsize)
            ax.set_ylabel(r'$\phi  \rightarrow$', fontsize=self.pltaxisfontsize)
            ax.set_zlabel(r'$\phi_2  \rightarrow$', fontsize=self.pltaxisfontsize, rotation=270)
        
            plt.tick_params(axis='both', which='major', labelsize=14)
            ax.zaxis.set_major_locator(LinearLocator(10))
            # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            plt.grid();plt.tight_layout();plt.show()
          
            
    @pyqtSlot()
    def ShowPERIDIST(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            I = EVOLVE.I
            m = EVOLVE.m
            n = EVOLVE.n
            p = EVOLVE.p
            label_img = label(I)
            regions = regionprops(label_img)
            count = 0
            peri = np.zeros(p)
            for props in regions:
                peri[count] = props.perimeter
                count = count + 1
                        
            plt.figure(ms + " Perimeter Distribution")
            plt.hist(peri, bins=20 , normed=1, color='green', alpha=0.8)
            plt.xlabel(r"$Perimeter\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Perimeter\; Probability \;Density\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.grid();plt.tight_layout();plt.show()

    @pyqtSlot()
    def ShowDISTNC(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            EVOLVE = getattr(self, myevolve)
            X = EVOLVE.X
            Y = EVOLVE.Y
            I = EVOLVE.I
            p = EVOLVE.p
            m = EVOLVE.m
            n = EVOLVE.n
            Xc, Yc = am.centroids(I, m, n, p)
            dist = np.sqrt((X - Xc) ** 2 + (Y - Yc) ** 2)
            plt.figure(ms + " N-C Distance Distribution")
            
            low = dist.min()
            high = dist.max()
            #print low, high, dist.mean(), dist.var()
            binS = np.linspace(low, high, 25)
            plt.hist(dist, normed=1, bins=binS)
            plt.xlabel(r"$X\;axis\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Y\;axis\rightarrow$", fontsize=self.pltaxisfontsize)
            #plt.legend(str(float(dist.mean())), loc='upper center')
            plt.grid();plt.tight_layout();plt.show()
            
    
    def find_confidence_interval(self, x, pdf, confidence_level):
        return pdf[pdf > x].sum() - confidence_level

    def density_contour(self, xdata, ydata, nbins_x, nbins_y, ax=None, **contour_kwargs):
        """ Create a density contour plot.
        Parameters
        ----------
        xdata : numpy.ndarray
        ydata : numpy.ndarray
        nbins_x : int
            Number of bins along x dimension
        nbins_y : int
            Number of bins along y dimension
        ax : matplotlib.Axes (optional)
            If supplied, plot the contour to this axis. Otherwise, open a new figure
        contour_kwargs : dict
            kwargs to be passed to pyplot.contour()
        """
    
        H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x, nbins_y), normed=True)
        x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1, nbins_x))
        y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y, 1))
    
        pdf = (H * (x_bin_sizes * y_bin_sizes))
    
        one_sigma = so.brentq(self.find_confidence_interval, 0., 1., args=(pdf, 0.68))
        two_sigma = so.brentq(self.find_confidence_interval, 0., 1., args=(pdf, 0.95))
        three_sigma = so.brentq(self.find_confidence_interval, 0., 1., args=(pdf, 0.99))
        levels = [one_sigma, two_sigma, three_sigma]
    
        X, Y = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
        Z = pdf.T
    
        if ax == None:
            contour = plt.contour(X, Y, Z, levels=levels, origin="lower", **contour_kwargs)
        else:
            contour = ax.contour(X, Y, Z, levels=levels, origin="lower", **contour_kwargs)
    
        return contour
    
    
    @pyqtSlot()
    def ShowLNNCDD(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            EVOLVE = getattr(self, myevolve)
            X = EVOLVE.X
            Y = EVOLVE.Y
            I = EVOLVE.I
            p = X.shape[0]
            m = I.shape[0]
            n = I.shape[1]
            # Xc, Yc = am.centroids(I, m, n, p)
            # dist = np.sqrt((X - Xc) ** 2 + (Y - Yc) ** 2)
            area = am.areadistribution(EVOLVE.I)
            # sigma, loc, mu = stats.lognorm.fit(dist, floc=0)
            mu, std = stats.norm.fit(area)
            #print mu, std
           
           
           
           
           
            # xmin = dist.min()
            # xmax = dist.max()
            # print (xmin, xmax)
            # x = np.linspace(xmin, xmax, 100)
            # pdf = stats.lognorm.pdf(x, sigma, scale=mu)
            # plt.plot(x, pdf, 'k')
            # plt.hist(dist, normed=1, bins=99)
            # plt.grid();plt.tight_layout();plt.show()
            # X = np.zeros(p)
            # for i in range(p):
                # X[i] = (dist[i] - xmin) / (xmax - xmin)
            
            # a, b, c, d = stats.beta.fit(dist)
            # print (a, b, c, d)
            
            # x = np.linspace(0,1)
            # x = np.linspace(stats.beta.ppf(0.01, a, b),stats.beta.ppf(0.99, a, b), 100)

            
            plt.hist(area, normed=1, bins=25);plt.grid();plt.tight_layout();plt.show()
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
            plt.tight_layout();plt.show()    
            # plt.plot(x, stats.beta.pdf(x, a, b));plt.grid();plt.tight_layout();plt.show()
            
    @pyqtSlot()
    def ShowANGLENC(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            X = EVOLVE.X
            Y = EVOLVE.Y
            I = EVOLVE.I
            p = X.shape[0]
            m = I.shape[0]
            n = I.shape[1]
            Xc, Yc = am.centroids(I, m, n, p)
            angle = np.arctan2(Y - Yc, X - Xc)
            plt.figure(ms + "N-C Displacement Orientation Distribution")
            plt.hist(angle, normed=1)
            plt.xlabel(r"$Nuclei-Centroid\; Vector\;Orientation (\theta) \rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Probability\;Density\; of \;\theta \rightarrow$", fontsize=self.pltaxisfontsize)
            plt.grid();plt.tight_layout();plt.show()
    
    
    
    @pyqtSlot()
    def ShowNND(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            I = EVOLVE.I
            nei = am.neighbors(I)
            plt.figure(ms + " Number of Edges Distribution")
            plt.hist(nei, normed=0, bins=25, color='green', alpha=0.8)
            plt.xlabel(r"$Number\; of\; Edges \rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Relative \; Fraction \rightarrow$", fontsize=self.pltaxisfontsize)
            plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            plt.grid();plt.tight_layout();plt.show()


    @pyqtSlot(QtCore.QPoint)
    def CMViewTexture(self, point):
        menu = QtGui.QMenu()
        actionCLEARALL = menu.addAction("Delete All Textures")
        actionDELSEL = menu.addAction("Delete Selected Texture")
        

        self.connect(actionDELSEL, SIGNAL("triggered()"), self, SLOT("DeleteThisTex()"))
        self.connect(actionCLEARALL, SIGNAL("triggered()"), self, SLOT("ClearAllTex()"))
        
        histmenu = QtGui.QMenu(menu)
        histmenu.setTitle("Plots")
        actionHISTPHI1 = histmenu.addAction("Phi1 Histogram")
        actionHISTPHI = histmenu.addAction("Phi Histogram")
        actionHISTPHI2 = histmenu.addAction("Phi2 Histogram")
        actionEULERSPACE = histmenu.addAction("Euler Space")
        actionANGLE = histmenu.addAction("Orientation Angle Distribution")
        actionAXIS = histmenu.addAction("Orientation Axis Distribution")
        actionPOLEFIG = histmenu.addAction("Pole Figure")
        actionMISANGLE = histmenu.addAction("Misorientation Angle Distribution")
        self.connect(actionHISTPHI1, SIGNAL("triggered()"), self, SLOT("HistPhi1()"))
        self.connect(actionHISTPHI, SIGNAL("triggered()"), self, SLOT("HistPhi()"))
        self.connect(actionHISTPHI2, SIGNAL("triggered()"), self, SLOT("HistPhi2()"))
        self.connect(actionEULERSPACE, SIGNAL("triggered()"), self, SLOT("EulerSpace()"))
        self.connect(actionPOLEFIG, SIGNAL("triggered()"), self, SLOT("PoleFigure()"))
        self.connect(actionANGLE, SIGNAL("triggered()"), self, SLOT("AngleDist()"))
        self.connect(actionAXIS, SIGNAL("triggered()"), self, SLOT("AxisDist()"))
        self.connect(actionMISANGLE, SIGNAL("triggered()"), self, SLOT("MISAngleDist()"))
        menu.addMenu(histmenu)
               
        menu.exec_(self.mapToGlobal(point))
        
    @pyqtSlot(QtCore.QPoint)
    def CMViewCANVAS(self, point):
        menu = QtGui.QMenu()
        actionCLEARALL = menu.addAction("Delete All Textures")
        actionDELSEL = menu.addAction("Delete Selected Texture")
        

        self.connect(actionDELSEL, SIGNAL("triggered()"), self, SLOT("DeleteThisTexCANVAS()"))
        self.connect(actionCLEARALL, SIGNAL("triggered()"), self, SLOT("ClearAllTexCANVAS()"))
        
        histmenu = QtGui.QMenu(menu)
        histmenu.setTitle("Plots")
        actionHISTPHI1 = histmenu.addAction("Phi1 Histogram")
        actionHISTPHI = histmenu.addAction("Phi Histogram")
        actionHISTPHI2 = histmenu.addAction("Phi2 Histogram")
        actionEULERSPACE = histmenu.addAction("Euler Space")
        actionPOLEFIG = histmenu.addAction("Plot Pole Figure")
        actionSMD = menu.addAction("Show Misorientation Distribution")
        self.connect(actionSMD, SIGNAL("triggered()"), self, SLOT("ShowMD()"))
        
        self.connect(actionHISTPHI1, SIGNAL("triggered()"), self, SLOT("HistPhi1CANVAS()"))
        self.connect(actionHISTPHI, SIGNAL("triggered()"), self, SLOT("HistPhiCANVAS()"))
        self.connect(actionHISTPHI2, SIGNAL("triggered()"), self, SLOT("HistPhi2CANVAS()"))
        self.connect(actionEULERSPACE, SIGNAL("triggered()"), self, SLOT("EulerSpaceCANVAS()"))
        self.connect(actionPOLEFIG, SIGNAL("triggered()"), self, SLOT("PoleFigureCANVAS()"))
        menu.addMenu(histmenu)
               
        menu.exec_(self.mapToGlobal(point))
    
    @pyqtSlot()
    def DeleteThisTexCANVAS(self):
        index, mytexture, ms = self.selectedCANVAS()
        if index != []:
            delattr(self, mytexture)    
            self.canvasmodel.beginRemoveRows(index.parent(), 0, 1)
            self.updateCANVASView()
        else:
            pass
    @pyqtSlot()
    def ClearAllTexCANVAS(self):
        number = self.CANVAScount
        for i in range(0, 1 + number):
            atb1 = 'GTC' + str(i)
            atb2 = 'URTC' + str(i)
            atb3 = 'FTC' + str(i)
            atb4 = 'STC' + str(i)  

            if  hasattr(self, atb1) == 1:
                A = getattr(self, atb1)
                makecanvas = getattr(A, "makecanvas")
                if makecanvas == 1:
                    delattr(self, atb1)
                    print("DELETED_" + atb1)
            elif  hasattr(self, atb2) == 1:
                A = getattr(self, atb2)
                makecanvas = getattr(A, "makecanvas")
                if makecanvas == 1:
                    delattr(self, atb2)
                    print("DELETED_" + atb2)
            elif  hasattr(self, atb3) == 1:
                A = getattr(self, atb3)
                makecanvas = getattr(A, "makecanvas")
                if makecanvas == 1:
                    delattr(self, atb3)
                    print("DELETED_" + atb3)
            elif  hasattr(self, atb4) == 1:
                A = getattr(self, atb4)
                makecanvas = getattr(A, "makecanvas")
                if makecanvas == 1:
                    delattr(self, atb4)
                    print("DELETED_" + atb4)
        
        self.updateCANVASView()
        self.CANVAScount = 0
        
    @pyqtSlot()
    def HistPhi1CANVAS(self):
        index, mytexture, ms = self.selectedCANVAS()
        if index != []:
            self.getpltoptions()
            Texture = getattr(self, mytexture)
            EUL = Texture.EUL
            m = EUL.shape[0]
            n = EUL.shape[1]
            Eul = EUL.reshape(m * n, 3)
            plt.hist(Eul[:, 0], normed=1, bins=50)
            plt.xlabel(r"$\phi_1\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Probability \;Density\; of\; \phi_1\rightarrow$", fontsize=self.pltaxisfontsize)           
            plt.grid();plt.tight_layout();plt.show()
    @pyqtSlot()
    def HistPhiCANVAS(self):
        index, mytexture, ms = self.selectedCANVAS()
        if index != []:
            Texture = getattr(self, mytexture)
            EUL = Texture.EUL
            m = EUL.shape[0]
            n = EUL.shape[1]
            Eul = EUL.reshape(m * n, 3)
            plt.hist(Eul[:, 1], normed=1, bins=50)
            plt.xlabel(r"$\phi\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Probability \;Density\; of\; \phi\rightarrow$", fontsize=self.pltaxisfontsize)            
            plt.grid();plt.tight_layout();plt.show()
    @pyqtSlot()
    def HistPhi2CANVAS(self):
        index, mytexture, ms = self.selectedCANVAS()
        if index != []:
            Texture = getattr(self, mytexture)
            EUL = Texture.EUL
            m = EUL.shape[0]
            n = EUL.shape[1]
            Eul = EUL.reshape(m * n, 3)
            plt.hist(Eul[:, 2], normed=1, bins=50)
            plt.xlabel(r"$\phi_2\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Probability \;Density\; of\; \phi_2\rightarrow$", fontsize=self.pltaxisfontsize)            
            plt.grid();plt.tight_layout();plt.show()
    
    @pyqtSlot()
    def EulerSpaceCANVAS(self):
        index, mytexture, ms = self.selectedCANVAS()
        if index != []:
            Texture = getattr(self, mytexture)
            EUL = Texture.EUL * 180 / pi
            m = EUL.shape[0]
            n = EUL.shape[1]
            Eul = EUL.reshape(m * n, 3)
            
            figname = ms + " Euler Space"
            if plt.fignum_exists(figname):
                plt.close(figname)
            fig = plt.figure(figname)
            
            ax = fig.gca(projection='3d')
            ax.scatter(Eul[:, 0], Eul[:, 1], Eul[:, 2], c='b', marker='.')
            ax.set_xlim(0, 360)
            ax.set_ylim(0, 180)
            ax.set_zlim(0, 360)                       
            ax.set_aspect('auto', 'datalim')
            ax.set_xlabel(r'$\phi_1  \rightarrow$', fontsize=self.pltaxisfontsize)
            ax.set_ylabel(r'$\phi  \rightarrow$', fontsize=self.pltaxisfontsize)
            ax.set_zlabel(r'$\phi_2  \rightarrow$', fontsize=self.pltaxisfontsize, rotation=270)
                 
            plt.tick_params(axis='both', which='major', labelsize=14)
            ax.zaxis.set_major_locator(LinearLocator(10)) 
            
            # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            plt.grid();plt.tight_layout();plt.show()
    @pyqtSlot()
    def PoleFigureCANVAS(self):
        sqt = np.sqrt(2)
        index, mytexture, ms = self.selectedCANVAS()
        if index != []:
            Texture = getattr(self, mytexture)
            EUL = Texture.EUL
            m = EUL.shape[0]
            n = EUL.shape[1]
            Eul = EUL.reshape(m * n, 3)
            XPx, YPx = self.polefig(Eul, [0, 0, 1])
            XPy, YPy = self.polefig(Eul, [1, 1, 0])
            XPz, YPz = self.polefig(Eul, [1, 1, 1])
            figname = ms + " Pole Figures"
            if plt.fignum_exists(figname):
                plt.close(figname)
            fig = plt.figure(figname)
            axisx = fig.add_subplot(1, 3, 1)
            axisy = fig.add_subplot(1, 3, 2)
            axisz = fig.add_subplot(1, 3, 3)
            axisx.plot(XPx, YPx, '.')
            axisy.plot(XPy, YPy, '.')
            axisz.plot(XPz, YPz, '.')
            
            axisx.set_xlim(-sqt, +sqt)
            axisx.set_ylim(-sqt, +sqt)
            axisy.set_xlim(-sqt, +sqt)
            axisy.set_ylim(-sqt, +sqt)
            axisz.set_xlim(-sqt, +sqt)
            axisz.set_ylim(-sqt, +sqt)
            axisx.set_aspect("equal")
            axisy.set_aspect("equal")
            axisz.set_aspect("equal")
      
            circle = plt.Circle((0, 0), sqt, color='cyan')
            axisx.add_artist(circle)
            circle = plt.Circle((0, 0), sqt, color='cyan')
            axisy.add_artist(circle)
            circle = plt.Circle((0, 0), sqt, color='cyan')
            axisz.add_artist(circle)
            
            axisx.set_title(r"$[0 0 1]$", fontsize=25)
            axisy.set_title(r"$[1 1 0]$", fontsize=25)
            axisz.set_title(r"$[1 1 1]$", fontsize=25)
            axisx.get_xaxis().set_visible(False)
            axisx.get_yaxis().set_visible(False)
            axisy.get_xaxis().set_visible(False)
            axisy.get_yaxis().set_visible(False)
            axisz.get_xaxis().set_visible(False)
            axisz.get_yaxis().set_visible(False)
            
            plt.grid();plt.tight_layout();plt.show()
    
    @pyqtSlot(QtCore.QPoint)
    def CMTextureBlend(self, point):
        menu = QtGui.QMenu()
        actionClearTextureList = menu.addAction("Clear List")
        self.connect(actionClearTextureList, SIGNAL("triggered()"), self, SLOT("ClearTextureList()"))
        menu.exec_(self.mapToGlobal(point))
            
    @pyqtSlot()
    def ClearTextureList(self):
        self.tableTextureBlend.setRowCount(0)
        self.tableTextureBlend.setColumnCount(0)
        
    
    @pyqtSlot()
    def ANGConvert(self):
        index, mymswt, ms = self.selectedMSWT()
        if index != []:
            A = getattr(self, mymswt)
            I = A.I
            m = A.m
            n = A.n
            EULER = A.EULER
            self.writeang(I, EULER, m, n)            
        else:
            pass
    
    def line_prepender(self, filename, line):
        with open(filename, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(line.rstrip('\r\n') + '\n' + content)
    
    
    def writeang(self, I, EULER, m, n) :

        MN = m * n

        RESTLIST = np.zeros((MN, 5))
        RESTLIST[0:, 0] = 255
        RESTLIST[0:, 1] = 1
        RESTLIST[0:, 2] = 1
        XYLIST = np.zeros((m * n, 2))
        EULERLIST = np.zeros((m * n, 3)) 

        for label in range(MN):
            i = label % m
            j = int(label / m)
            curr = I[i, j] - 1
            XYLIST[label, 0] = i
            XYLIST[label, 1] = j 
            EULERLIST[label, 0] = EULER[curr, 0]
            EULERLIST[label, 1] = EULER[curr, 1]
            EULERLIST[label, 2] = EULER[curr, 2]
              
        C = np.concatenate((EULERLIST, XYLIST), axis=1)
        D = np.concatenate((C, RESTLIST), axis=1)
        np.savetxt('test.ang', D, delimiter='\t', fmt='%0.4f')
        
        def line_prepender(filename, line):
            with open(filename, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(line.rstrip('\r\n') + '\n' + content)
        
        
        line_prepender('test.ang', '# NumberFamilies\t0')
        line_prepender('test.ang', '# LatticeConstants\t 1\t 1\t 1\t 90.000\t 90.000\t 90.000')
        line_prepender('test.ang', '# Symmetry\t 43')
        line_prepender('test.ang', '# Formula\t random')
        line_prepender('test.ang', '# MaterialName\t random')
          
    
    @pyqtSlot()
    def ShowMD(self):
        index, mymswt, ms = self.selectedMSWT()
        if index != []:
            mymswt = getattr(self, mymswt)
            I = getattr(mymswt, 'I')
            Euler = getattr(mymswt, 'EULER')
            Rot = conv.zxz_to_rm(Euler)
            theta = at.disorientation(I, Rot, 1 , 0)
            theta = theta * 180 / pi
            theta = np.ma.masked_equal(theta, 0)
            plt.figure(ms + " Misorientation Distribution")
            plt.hist(theta.compressed(), bins=50, normed=1, color='green', alpha=0.8)
            plt.xlabel(r"$MIsorientation\;Angle\;\gamma\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Probability \;Density\; of\; \gamma\rightarrow$", fontsize=self.pltaxisfontsize) 
            plt.grid();plt.tight_layout();plt.show()
            
    
    def AssignTextureToMS(self):
        PerctList = []
        EulList = []
        MSWT = MSWithTexture()
        TEXCount = 0
        
        c = self.tabWidget_howtoassign.currentIndex()
        MS = str(self.comboBox_MSList.currentText())
        if MS == "":
            QtGui.QMessageBox.about(self, "Error!", "No Microstructure Found")
            return 0
        else:
            EVOLVE = "EVOLVE" + MS.replace("MS", "")
            if hasattr(self, EVOLVE):            
                ev = getattr(self, EVOLVE)
                I = getattr(ev, "I")
                setattr(MSWT, "I", I)
                MSWT.m = I.shape[0]
                MSWT.n = I.shape[1]
                MSWT.p = np.max(I)
        
        if c == 0:
            rows = self.tableTextureBlend.rowCount()
            if rows == 0:
                QtGui.QMessageBox.about(self, "Error!", "No Texture Found")
                return 0
                
            
            for row in range(rows):
                texname = self.tableTextureBlend.item(row, 0).text()
                texperctstring = self.tableTextureBlend.item(row, 1).text()
                
                
                if  hasattr(self, texname):
                    EulList.append(getattr(getattr(self, texname), "Eul").T) 
                    
                else:
                    QtGui.QMessageBox.about(self, "Error!", texname + " Does Not Exist Anymore") 
                    return 0
    
                if texperctstring.isdigit():
                    PerctList.append(float(texperctstring))
               
                else:
                    QtGui.QMessageBox.about(self, "Error!", "Invalid Texture Percentage Value !")
                    return 0
    
                TEXCount = TEXCount + 1
                   
            MSWT.Frac = np.hstack(PerctList) / 100.0
            MSWT.Eul = np.dstack(EulList).T
            MSWT.TEXCount = TEXCount
            self.MSWT = MSWT
            
            MSWT.Assign_AS_RELATIVE_FRACTION(self.MSWT)
            MSWT.COLOR = col.EulToCol(MSWT.I, MSWT.EULER, MSWT.p)
            
            
            if self.checkBox_KSMSWT.isChecked():
                self.KSMSWT = 1
                self.MSWTcount = self.MSWTcount + 1
            else:
                self.KSMSWT = 0
                self.MSWTcount = 0
            currentmswt = "MSWT" + str(self.MSWTcount)
            setattr(self, currentmswt , self.MSWT)
            delattr(self, "MSWT")
                
            cv2.namedWindow(currentmswt, cv2.WINDOW_NORMAL)
            cv2.imshow(currentmswt, getattr(getattr(self, currentmswt), "COLOR"))
            self.updateMSWTView()
            
        
    def selectedTEXTURE(self):
        # self.tvURT.model().clear()
        index = self.tvTEX.selectedIndexes()
        if index != []:
            item = self.texmodel.itemFromIndex(index[0]).text()
            for i in range(0, self.TEXTUREcount + 1):
                atb1 = 'GT' + str(i)
                atb2 = 'URT' + str(i)
                atb3 = 'FT' + str(i)
                atb4 = 'ST' + str(i)  
        
                if  hasattr(self, atb1) == 1:
                    if item == atb1:
                        print("SELECTED_" + atb1)
                        return index[0], atb1, str(item)
                    
                elif  hasattr(self, atb2) == 1:
                    if item == atb2:
                        print("SELECTED_" + atb2)
                        return index[0], atb2, str(item)
                    
                elif  hasattr(self, atb3) == 1:
                    if item == atb3:
                        print("SELECTED_" + atb3)
                        return index[0], atb3, str(item)
                    
                elif  hasattr(self, atb4) == 1:
                    if item == atb4:
                        print("SELECTED_" + atb4)
                        return index[0], atb4, str(item)

        else:
                return [], [], []
            
    def selectedCANVAS(self):
        index = self.tvCANVAS.selectedIndexes()
        if index != []:
            item = self.canvasmodel.itemFromIndex(index[0]).text()
            for i in range(0, self.CANVAScount + 1):
                atb1 = 'GTC' + str(i)
                atb2 = 'URTC' + str(i)
                atb3 = 'FTC' + str(i)
                atb4 = 'STC' + str(i)  
        
                if  hasattr(self, atb1) == 1:
                    if item == atb1:
                        print("SELECTED_" + atb1)
                        return index[0], atb1, str(item)
                    
                elif  hasattr(self, atb2) == 1:
                    if item == atb2:
                        print("SELECTED_" + atb2)
                        return index[0], atb2, str(item)
                    
                elif  hasattr(self, atb3) == 1:
                    if item == atb3:
                        print("SELECTED_" + atb3)
                        return index[0], atb3, str(item)
                    
                elif  hasattr(self, atb4) == 1:
                    if item == atb4:
                        print("SELECTED_" + atb4)
                        return index[0], atb4, str(item)

        else:
                return [], [], []
    
    
   
    @pyqtSlot()
    def HistPhi1(self):
        index, mytexture, ms = self.selectedTEXTURE()
        if index != []:
            self.getpltoptions()
            TextureURT = getattr(self, mytexture)
            Eul = TextureURT.Eul
            figname = ms + " Phi1 Histogram"
            if plt.fignum_exists(figname):
                plt.close(figname)
            plt.figure(figname)
            plt.hist(Eul[:, 0], normed=1, bins=50)
            plt.xlim(0, 2 * pi)
            plt.grid();plt.tight_layout();plt.show()
    
    @pyqtSlot()
    def HistPhi(self):
        index, mytexture, ms = self.selectedTEXTURE()
        if index != []:
            self.getpltoptions()
            TextureURT = getattr(self, mytexture)
            Eul = TextureURT.Eul
            figname = ms + " Phi Histogram"
            if plt.fignum_exists(figname):
                plt.close(figname)
            plt.figure(figname)
            plt.hist(Eul[:, 1], normed=1, bins=50)
            plt.xlim(0, pi)
            plt.grid();plt.tight_layout();plt.show()
            
    @pyqtSlot()
    def HistPhi2(self):
        index, mytexture, ms = self.selectedTEXTURE()
        if index != []:
            self.getpltoptions()
            TextureURT = getattr(self, mytexture)
            Eul = TextureURT.Eul
            figname = ms + " Phi2 Histogram"
            if plt.fignum_exists(figname):
                plt.close(figname)
            plt.figure(figname)
            plt.hist(Eul[:, 2], normed=1, bins=50)
            plt.xlim(0, 2 * pi)
            plt.grid();plt.tight_layout();plt.show()
    
            
    @pyqtSlot()
    def AxisDist(self):
        index, mytexture, ms = self.selectedTEXTURE()
        if index != []:
            self.getpltoptions()
            Texture = getattr(self, mytexture)
            Eul = Texture.Eul
            
            figname = ms + " Axis Distribution Over Unit Sphere"
            if plt.fignum_exists(figname):
                plt.close(figname)
            fig = plt.figure(figname)
            
            axis = fig.gca(projection='3d')
                       
            Axis, _ = conv.zxz_to_aap(Eul)
            axis.scatter(Axis[:, 0], Axis[:, 1], Axis[:, 2], c='b', marker='.')
            axis.set_xlim(-1, 1)
            axis.set_ylim(-1, 1)
            axis.set_zlim(-1, 1)
            axis.set_xlabel(r"$X\;axis\rightarrow$", fontsize=self.pltaxisfontsize)
            axis.set_ylabel(r"$Y\;axis\rightarrow$", fontsize=self.pltaxisfontsize)
            axis.set_zlabel(r"$Z\;axis\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.grid();plt.tight_layout();plt.show()
            
    @pyqtSlot()
    def AngleDist(self):
        index, mytexture, ms = self.selectedTEXTURE()
        if index != []:
            self.getpltoptions()
            Texture = getattr(self, mytexture)
            Eul = Texture.Eul
            
            figname = ms + " Angle Distribution"
            if plt.fignum_exists(figname):
                plt.close(figname)
            plt.figure(figname)

            _, Angle = conv.zxz_to_aap(Eul)
            plt.hist(Angle[np.isfinite(Angle)], bins=50, normed=1)
            plt.xlabel(r"$Angle(\gamma)\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Probability\;Density\;of\;\gamma\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.grid();plt.tight_layout();plt.show()
            
    @pyqtSlot()
    def MISAngleDist(self):
        index, mytexture, ms = self.selectedTEXTURE()
        if index != []:
            self.getpltoptions()
            Texture = getattr(self, mytexture)
            Eul = Texture.Eul
            
            figname = ms + " Misorientation Angle Distribution"
            if plt.fignum_exists(figname):
                plt.close(figname)
            plt.figure(figname)
            Rot = conv.zxz_to_rm(Eul)
            theta = at.disorientation(np.zeros((1,1),dtype=np.int64), Rot, 4 , 0)
            theta = theta * 180 / pi
            theta = np.ma.masked_equal(theta, 0)
            plt.hist(theta.compressed(), bins=50, normed=1, color='green', alpha=0.8)
            plt.xlabel(r"$Misorientation\;Angle\;\gamma\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Probability \;Density\; of\; \gamma\rightarrow$", fontsize=self.pltaxisfontsize) 
            plt.grid();plt.tight_layout();plt.show()

    
    
    
    @pyqtSlot()        
    def EulerSpace(self):    
        index, mytexture, ms = self.selectedTEXTURE()
        if index != []:
            Texture = getattr(self, mytexture)
            Eul = Texture.Eul * 180 / pi
            figname = ms + " Euler Space"
            if plt.fignum_exists(figname):
                plt.close(figname)
            fig = plt.figure(figname)
            self.getpltoptions()
            ax = fig.gca(projection='3d')
            ax.scatter(Eul[:, 0], Eul[:, 1], Eul[:, 2], c='b', marker='.')
            ax.set_xlim(0, 360)
            ax.set_ylim(0, 180)
            ax.set_zlim(0, 360)                       
            ax.set_aspect('auto', 'datalim')
            ax.set_xlabel(r'$\phi_1  \rightarrow$', fontsize=self.pltaxisfontsize)
            ax.set_ylabel(r'$\phi  \rightarrow$', fontsize=self.pltaxisfontsize)
            ax.set_zlabel(r'$\phi_2  \rightarrow$', fontsize=self.pltaxisfontsize, rotation=270)
                 
            # plt.tick_params(axis='both', which='major', labelsize=14)

            plt.grid();plt.tight_layout();plt.show()

    
    @pyqtSlot()
    def DeleteThisTex(self):
        index, mytexture, ms = self.selectedTEXTURE()
        if index != []:
            delattr(self, mytexture)    
            self.texmodel.beginRemoveRows(index.parent(), 0, 1)
            self.updateTEXTUREView()
        else:
            pass
        
    
    @pyqtSlot()    
    def ClearAllTex(self):
        number = self.TEXTUREcount
        for i in range(0, 1 + number):
            atb1 = 'GT' + str(i)
            atb2 = 'URT' + str(i)
            atb3 = 'FT' + str(i)
            atb4 = 'ST' + str(i)  

            if  hasattr(self, atb1) == 1:
                A = getattr(self, atb1)
                makecanvas = getattr(A, "makecanvas")
                if makecanvas == 0:
                    delattr(self, atb1)
                    print("DELETED_" + atb1)
            elif  hasattr(self, atb2) == 1:
                A = getattr(self, atb2)
                makecanvas = getattr(A, "makecanvas")
                if makecanvas == 0:
                    delattr(self, atb2)
                    print("DELETED_" + atb2)
            elif  hasattr(self, atb3) == 1:
                A = getattr(self, atb3)
                makecanvas = getattr(A, "makecanvas")
                if makecanvas == 0:
                    delattr(self, atb3)
                    print("DELETED_" + atb3)
            elif  hasattr(self, atb4) == 1:
                A = getattr(self, atb4)
                makecanvas = getattr(A, "makecanvas")
                if makecanvas == 0:
                    delattr(self, atb4)
                    print("DELETED_" + atb4)
        
        self.updateTEXTUREView()
        self.TEXTUREcount = 0
    
    
    
    @pyqtSlot()
    def ShowCAW(self):
        plt.close("all")
        cv2.destroyAllWindows()
        
    
            
    
    @pyqtSlot()
    def ShowGAD(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            area = am.areadistribution(EVOLVE.I)
            plt.figure(ms + " Grain Area Distribution")
            plt.hist(area, bins=50 , normed=1, color='green', alpha=0.8)
            plt.xlabel(r"$Grain\;Area \rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Probability\;Density \rightarrow$", fontsize=self.pltaxisfontsize)
            plt.grid();plt.tight_layout();plt.show()
    
    
    @pyqtSlot()
    def ShowBOUND(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            I = np.array(EVOLVE.I, dtype=np.float64)
            plt.figure(ms + " Edges")
            B = sobel(I)
            B[B > 0] = 1
            B = np.array(B, dtype=np.uint8)
            kern = np.ones((3,3),dtype=np.uint8)            
            plt.imshow(np.invert(cv2.dilate(B,kern)), cmap=self.colmaps)
            plt.grid();plt.tight_layout();plt.show()
    
    @pyqtSlot()
    def ShowDIA(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            I = EVOLVE.I
            p = EVOLVE.p
            label_img = label(I)
            regions = regionprops(label_img)
            dia = np.zeros(p)
            count = 0
            for props in regions:
                dia[count] = props.equivalent_diameter 
                count = count + 1

            plt.figure(ms + " Grain Diameter Distribution")
            plt.hist(dia, bins=50 , normed=1, color='green', alpha=0.8)
            plt.xlabel(r"$Grain\;Diameter\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Probability\;Density\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.grid();plt.tight_layout();plt.show()
    
    @pyqtSlot()
    def ShowTRIP(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            I = EVOLVE.I
            m = EVOLVE.m
            n = EVOLVE.n
            p = EVOLVE.n
            X, Y = am.triplepoints(I, m, n, p)
            plt.figure(ms + " Triple Points")
            plt.title(ms + " Triple Points")
            plt.plot(Y, X, 'o', color='black', markerfacecolor=(0.5, 0.5, 0.5, 1))
            plt.imshow(EVOLVE.Col[:, :, ::-1], origin='upper', interpolation='none', cmap=self.colmaps)
            # plt.xlabel(r"$X\;axis\rightarrow$", fontsize=self.pltaxisfontsize)
            # plt.ylabel(r"$Y\;axis\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.grid();plt.tight_layout();plt.show()
            
    @pyqtSlot()
    def ShowShapeFactor(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            I = EVOLVE.I
            p = EVOLVE.p
            label_img = label(I)
            regions = regionprops(label_img)
            fact = np.zeros(p)
            count = 0
            for props in regions:
                mya = props.area
                myp = props.perimeter
                fact[count] = myp ** 2 / (4 * pi * mya)
                count = count + 1
            
            
            plt.figure(ms + " Shape Factor")
            fact = np.ma.masked_less(fact, 1)
            plt.hist(fact.compressed(), bins=50, normed=1, color='green', alpha=0.8)
            plt.xlabel(r"$Shape\;Factor\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Probability\;Density\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.grid();plt.tight_layout();plt.show()
    
    
    @pyqtSlot()
    def ShowEccentricityDIST(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            I = EVOLVE.I
            p = EVOLVE.p
            label_img = label(I)
            regions = regionprops(label_img)
            ecc = np.zeros(p)
            count = 0
            for props in regions:
                ecc[count] = props.eccentricity
                count = count + 1

            plt.figure(ms + " Eccentricity")
            plt.hist(ecc, bins=50 , normed=1, color='green', alpha=0.8)
            plt.xlabel(r"$Eccentricity\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Probability Density\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.grid();plt.tight_layout();plt.show()
            
    
    @pyqtSlot()
    def ShowEllOriDIST(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            I = EVOLVE.I
            p = EVOLVE.p
            label_img = label(np.flipud(I))
            regions = regionprops(label_img)
            ori = np.zeros(p)
            count = 0
            for props in regions:
                ori[count] = props.orientation               
                count = count + 1
            binS = 5 * np.arange(37)
            plt.figure(ms + " Elliptical Orientation DIstribution")
            plt.hist((pi / 2 - ori) * (180 / pi), bins=binS , normed=1, color='green', alpha=0.8)
            plt.xlabel(r"$Ellipse\;Orientation\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.ylabel(r"$Probability\;Density\rightarrow$", fontsize=self.pltaxisfontsize)
            plt.grid();plt.tight_layout();plt.show()
    
    
    
    @pyqtSlot()
    def ShowNS(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            X = EVOLVE.X
            Y = EVOLVE.Y
            m = EVOLVE.m
            n = EVOLVE.n
            plt.figure(ms + " Nucleation Sites")
            plt.imshow(EVOLVE.Col[:, :, ::-1], interpolation='none', cmap=self.colmaps)
            plt.plot(Y, X, '.',color='black', markerfacecolor=(1, 1, 0, 1))
            plt.xlim([0, n])
            plt.ylim([n, 0])
            plt.grid();plt.tight_layout();plt.show()
    
    @pyqtSlot()
    def ShowSCEN(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            EVOLVE = getattr(self, myevolve)
            I = EVOLVE.I
            p = EVOLVE.p
            m = EVOLVE.m
            n = EVOLVE.n
            Xc, Yc = am.centroids(I, m, n, p)
            plt.figure(ms + " Grain Centroids")
            plt.imshow(EVOLVE.Col[:, :, ::-1], interpolation='none', cmap=self.colmaps)
            plt.plot(Yc, Xc, '.',color='black', markerfacecolor=(1, 1, 0, 1))
            plt.xlim([0, n])
            plt.ylim([n, 0])
            plt.grid();plt.tight_layout();plt.show()
          
    
    @pyqtSlot()
    def ClearAllMS(self):
        number = self.EVOLVEcount
        for i in range(0, 1 + number):
            mystring = 'EVOLVE' + str(i)
            if hasattr(self, mystring):
                delattr(self, mystring)
                print("DELETED_MS" + str(i))
        self.updateMSView()
        self.EVOLVEcount = 0
        
    @pyqtSlot()
    def ClearAllMSWT(self):
        number = self.MSWTcount
        for i in range(0, 1 + number):
            mystring = 'MSWT' + str(i)
            if hasattr(self, mystring):
                delattr(self, mystring)
                print("DELETED_MSWT" + str(i))
        self.updateMSWTView()
        self.MSWTcount = 0
        
        
               
    @pyqtSlot()
    def DeleteThisMS(self):
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            delattr(self, myevolve)    
            self.msmodel.beginRemoveRows(index.parent(), 0, 1)
            self.updateMSView()
        else:
            self.choosemsfirst()
    @pyqtSlot()    
    def DeleteThisMSWT(self):
        index, mymswt, ms = self.selectedMSWT()
        if index != []:
            delattr(self, mymswt)    
            self.mswtmodel.beginRemoveRows(index.parent(), 0, 1)
            self.updateMSWTView()
    
    @pyqtSlot()
    def ShowCOL(self):
        cv2.destroyWindow("Microstructure Evolution Random Color")
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            A = getattr(self, myevolve)
            figname = ms + " Random Colored"
            if plt.fignum_exists(figname):
                plt.close(figname)
            plt.figure(figname)
            plt.imshow(A.Col[:, :, ::-1], origin='upper', interpolation='none', cmap=self.colmaps)
            plt.tight_layout();plt.show()
      
        
    @pyqtSlot()
    def ShowOPENCV(self):
        cv2.destroyWindow("Microstructure Evolution Random Color")
        index, myevolve, ms = self.selectedEVOLVE()
        if index != []:
            self.getpltoptions()
            A = getattr(self, myevolve)
            cv2.namedWindow(ms + " Random Colored- OPENCV Frame", cv2.WINDOW_NORMAL)
            cv2.imshow(ms + " Random Colored- OPENCV Frame", A.Col)
            
        else:
            pass
        
    def ShowThisMSWT(self):
        index, mymswt, ms = self.selectedMSWT()
        if index != []:
            A = getattr(self, mymswt)
            cv2.namedWindow(mymswt, cv2.WINDOW_NORMAL)
            cv2.imshow(mymswt, A.COLOR)
        else:
            pass
    
   
    def selectedEVOLVE(self):
        index = self.tvMS.selectedIndexes()
        if index != []:
            item = self.msmodel.itemFromIndex(index[0]).text()
            for i in range(0, self.EVOLVEcount + 1):
                if item == "MS" + str(i):
                    myevolve = 'EVOLVE' + str(i)
                    return index[0], myevolve, str(item)
        else:
                return [], [], []
        
    def selectedMSWT(self):
        index = self.tvMSWT.selectedIndexes()
        if index != []:
            item = self.mswtmodel.itemFromIndex(index[0]).text()
            for i in range(0, self.MSWTcount + 1):
                mymswt = 'MSWT' + str(i)
                if  hasattr(self, mymswt) == 1:
                    if item == mymswt:
                        print("SELECTED_" + mymswt)
                        return index[0], mymswt, str(item)
               

        else:
                return [], [], []
        
    
    
    
    def grabInfoMS(self):
        EVOLVE = EvolutionFunctions()
        
        c1 = self.tabWidget_2D3D.currentIndex()        
        if c1 is 0:
            mycase = 'Evolve_2D'
            EVOLVE.m = int(self.dimX2D.text())
            EVOLVE.n = int(self.dimY2D.text())
            EVOLVE.framepause = int(self.framepause.text())
            if self.checkBox_pdelNxy.isChecked() is True:
                mytext = str(self.textEdit_pdelNxy.toPlainText())
                try:
                    EVOLVE.pdelNxy = lambdify('x,y', mytext, self.mydict)
                except:
                    print("ERROR")
                    self.setEnabled(True)
            else:
                EVOLVE.pdelNxy = []        
            
            if self.shouldwesave.isChecked():
                EVOLVE.sf = 1
                EVOLVE.fd = self.framepath.toPlainText()
    
            else:
                EVOLVE.sf = 0
                EVOLVE.fd = ''
    
            if self.animshow.isChecked():
                EVOLVE.sa = self.comboBox.currentIndex() + 1

            else:
                EVOLVE.sa = 0
            
            if self.checkBox_useseed.isChecked():
                EVOLVE.myseed = int(self.spinBox_myseed.text())
            else:
                EVOLVE.myseed = 0
            
            if self.checkBox_labelsort.isChecked():
                EVOLVE.labelsorted = 1
            else:
                EVOLVE.labelsorted = 0
                
            
                
            if self.checkBoxKSM.isChecked():
                self.KSM = 1
            else:
                self.KSM = 0
            
            c2 = self.tabWidget_2D_IsoAniso.currentIndex()
            if c2 is 0:
                mycase = mycase + '_Isotropic'
                c3 = self.tabWidget_2D_Iso_SiteConti.currentIndex()
                if c3 is 0:
                    mycase = mycase + '_SiteSaturated'
                else:
                    mycase = mycase + '_Continuous'
                    self.radioButton_norec.setChecked(True)                                
            else:
                mycase = mycase + '_Anisotropic'
                c3 = self.tabWidget_2D_Aniso_SiteConti.currentIndex()
                if c3 is 0:
                    mycase = mycase + '_SiteSaturated'
                    c4 = self.tabWidget_2D_Aniso_Site_NeighGeneric.currentIndex()
                    if c4 is 0:
                        mycase = mycase + '_NeighbourHoodBased'
                    else:
                        mycase = mycase + '_Generic'
                        c5 = self.tabWidget_2D_Aniso_Site_Generic_EllipDirText.currentIndex()
                        if c5 == 0:
                            mycase = mycase + '_Elliptical'
                        elif c5 == 1:
                            mycase = mycase + '_DirectionBased'
                        elif c5 == 2:
                            mycase = mycase + '_TextureBased'
                            c6 = self.tabWidget_2DTextureBased.currentIndex()
                            if c6 == 0:
                                mycase = mycase + '_Ellipsoidal'
                            elif c6 == 1:
                                mycase = mycase + '_UserDefined'
                
                elif c3 is 1:
                    mycase = mycase + '_Continuous'
                    self.radioButton_norec.setChecked(True)
                    c4 = self.tabWidget_2D_Aniso_Conti_NeighGeneric.currentIndex()
                    if c4 is 0:
                        mycase = mycase + '_NeighbourHoodBased'
                    else:
                        mycase = mycase + '_Generic'
                        c5 = self.tabWidget_2D_Aniso_Conti_Generic_EllipDirText.currentIndex()
                        if c5 == 0:
                            mycase = mycase + '_Elliptical'
                        elif c5 == 1:
                            mycase = mycase + '_DirectionBased'
                        elif c5 == 2:
                            mycase = mycase + '_TextureBased'
            
            if self.radioButton_norec.isChecked() is True:
                EVOLVE.asy = 0
            else:
                EVOLVE.asy = 1
            
        else:
            mycase = '3D'

        self.mycase = mycase
        
                
        # GRABBING INPUTS BASED ON THE CASE
        if self.mycase == 'Evolve_2D_Isotropic_SiteSaturated':
            
            EVOLVE.p = int(self.numP2DIsoSite.text())
            if self.checkBox_gr2D.isChecked() is True:
                mytext = str(self.textEdit_gr2D.toPlainText())
                EVOLVE.gr2D = lambdify('x,y,l', mytext, self.mydict)
            else:
                EVOLVE.gr2D = []
            
        elif self.mycase == 'Evolve_2D_Isotropic_Continuous':
            mytext = str(self.textEdit_NdotIso2D.toPlainText())
            EVOLVE.NdotIsoXY = lambdify('t', mytext, self.mydict)
            mytext = str(self.textEdit_GtIso2D.toPlainText())
            EVOLVE.GtIsoXY = lambdify('t', mytext, self.mydict)
            EVOLVE.fstop = float(self.doubleSpinBox_NustopIso2D.text()) / 100
            if self.checkBox_gr2D.isChecked() is True:
                mytext = str(self.textEdit_gr2D.toPlainText())
                EVOLVE.gr2D = lambdify('x,y,l', mytext, self.mydict)
            else:
                EVOLVE.gr2D = []
        
        elif self.mycase == 'Evolve_2D_Anisotropic_SiteSaturated_NeighbourHoodBased':
            EVOLVE.p = int(self.numP2DAnisoSite.text())
            if self.checkBox_gr2D_Aniso_Site_Neigh.isChecked() is True:
                mytext = str(self.textEdit_gr2D_Site_Neigh.toPlainText())
                EVOLVE.gr2D = lambdify('x,y,l', mytext, self.mydict)
            else:
                EVOLVE.gr2D = []
                
            if self.radioButton_N4Site.isChecked() is True:
                EVOLVE.neightype = 4
            elif self.radioButton_N6Site.isChecked() is True:
                EVOLVE.neightype = 6
            elif self.radioButton_N8Site.isChecked() is True:
                EVOLVE.neightype = 8
                
        elif self.mycase == 'Evolve_2D_Anisotropic_Continuous_NeighbourHoodBased':
            mytext = str(self.textEdit_Ndot_Aniso_Neigh.toPlainText())
            EVOLVE.Ndot = lambdify('t', mytext, self.mydict)
            mytext = str(self.textEdit_Gt_Aniso_Neigh.toPlainText())
            EVOLVE.Gt = lambdify('t', mytext, self.mydict)
            EVOLVE.fstop = float(self.doubleSpinBox_NustopAnisoNeigh.text()) / 100

            if self.checkBox_gr2D_Aniso_Conti_Neigh.isChecked() is True:
                mytext = str(self.textEdit_gr2D_aniso_conti_neigh.toPlainText())
                EVOLVE.gr2D = lambdify('x,y,l', mytext, self.mydict)
            else:
                EVOLVE.gr2D = []
                
            if self.radioButton_N4Conti.isChecked() is True:
                EVOLVE.neightype = 4
            elif self.radioButton_N6Conti.isChecked() is True:
                EVOLVE.neightype = 6
            elif self.radioButton_N8Conti.isChecked() is True:
                EVOLVE.neightype = 8
                
        elif self.mycase == 'Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical':
            EVOLVE.p = int(self.numP2DAnisoSite.text())
            EVOLVE.seq = self.comboBox_order.currentIndex() + 1
            if self.checkBox_thetarvector.isChecked() is True:
                mytext = str(self.textEdit_thetarvector.toPlainText())
                EVOLVE.Thetafunc = lambdify('x,y,l,R,adot', mytext, self.mydict)
            else:
                EVOLVE.Thetafunc = []
            if self.checkBox_Rrvector.isChecked() is True:
                mytext = str(self.textEdit_Rrvector.toPlainText())
                EVOLVE.Rfunc = lambdify('x,y,l,adot,theta', mytext, self.mydict)
            else:
                EVOLVE.Rfunc = []
                EVOLVE.R = float(self.doubleSpinBox_Rrvector.text())
            if self.checkBox_adotrvector.isChecked() is True:
                mytext = str(self.textEdit_adotrvector.toPlainText())
                EVOLVE.Adotfunc = lambdify('x,y,l,theta,R', mytext, self.mydict)
            else:
                EVOLVE.Adotfunc = []  
        elif self.mycase == 'Evolve_2D_Anisotropic_Continuous_Generic_Elliptical':
            
            mytext = str(self.textEdit_Ndot_Aniso_Ellip.toPlainText())
            EVOLVE.Ndot = lambdify('t', mytext, self.mydict)
            mytext = str(self.textEdit_Aniso_Conti_adott.toPlainText())
            EVOLVE.Gt = lambdify('t', mytext, self.mydict)            
            EVOLVE.fstop = float(self.doubleSpinBox_NustopAnisoEllip.text()) / 100
            EVOLVE.seq = self.comboBox_order2.currentIndex() + 1
            
            if self.checkBox_thetarvector_2.isChecked() is True:
                mytext = str(self.textEdit_thetarvector_2.toPlainText())
                EVOLVE.Thetafunc = lambdify('x,y,l,R,adot', mytext, self.mydict)
            else:
                EVOLVE.Thetafunc = []
            if self.checkBox_Rrvector_2.isChecked() is True:
                mytext = str(self.textEdit_Rrvector_2.toPlainText())
                EVOLVE.Rfunc = lambdify('x,y,l,adot,theta', mytext, self.mydict)
            else:
                EVOLVE.Rfunc = []
                EVOLVE.R = float(self.doubleSpinBox_Rrvector_2.text())
            if self.checkBox_adotrvector_2.isChecked() is True:
                mytext = str(self.textEdit_adotrvector_2.toPlainText())
                EVOLVE.Adotfunc = lambdify('x,y,l,theta,R', mytext, self.mydict)
            else:
                EVOLVE.Adotfunc = []
        
        elif self.mycase == 'Evolve_2D_Anisotropic_SiteSaturated_Generic_DirectionBased':
            
            EVOLVE.angularresolution = int(self.angresolution.text())
            EVOLVE.p = int(self.numP2DAnisoSite.text())
            
            mytext = str(self.textEdit_Gphi.toPlainText())
            EVOLVE.Gphi = lambdify('x,y,l', mytext, self.mydict)
            
            if self.checkBox_thetarvector_3.isChecked() is True:
                mytext = str(self.textEdit_thetarvector_3.toPlainText())
                EVOLVE.Thetafunc = lambdify('x,y,l', mytext, self.mydict)
            else:
                EVOLVE.Thetafunc = []
        
        
            if self.checkBox_Grvector.isChecked() is True:
                mytext = str(self.textEdit_Grvector.toPlainText())
                EVOLVE.Grvector = lambdify('x,y,l', mytext, self.mydict)
            else:
                EVOLVE.Grvector = []
        
        elif self.mycase == 'Evolve_2D_Anisotropic_SiteSaturated_Generic_TextureBased_Ellipsoidal':
            EVOLVE.p = int(self.numP2DAnisoSite.text())
            if self.checkBox_Rab.isChecked() is True:
                mytext = str(self.textEdit_Rab.toPlainText())
                EVOLVE.Rabfunc = lambdify('x,y,l', mytext, self.mydict)
            else:
                EVOLVE.Rabfunc = []
                EVOLVE.Rab = float(self.doubleSpinBox_Rab.text())
                
            if self.checkBox_Rac.isChecked() is True:
                mytext = str(self.textEdit_Rac.toPlainText())
                EVOLVE.Racfunc = lambdify('x,y,l', mytext, self.mydict)
            else:
                EVOLVE.Racfunc = []
                EVOLVE.Rac = float(self.doubleSpinBox_Rac.text())
                
            if self.checkBox_adotrvec.isChecked() is True:
                mytext = str(self.textEdit_adotrvec.toPlainText())
                EVOLVE.Adotfunc = lambdify('x,y,l', mytext, self.mydict)
            else:
                EVOLVE.Adotfunc = []
            
            
            PerctList = []
            EulList = []
            TEXCount = 0
            MSWT = MSWithTexture()    
            rows = self.tableCanvasBlend.rowCount()
            if rows == 0:
                QtGui.QMessageBox.about(self, "Error!", "No Texture Canvas Found")
                return 0
                
            
            for row in range(rows):
                texname = self.tableCanvasBlend.item(row, 0).text()
                texperctstring = self.tableCanvasBlend.item(row, 1).text()
                
                
                if  hasattr(self, texname):
                    EulList.append(getattr(getattr(self, texname), "EUL").T) 
                    
                else:
                    QtGui.QMessageBox.about(self, "Error!", texname + " Does Not Exist Anymore") 
                    return 0
    
                if texperctstring.isdigit():
                    PerctList.append(float(texperctstring))
               
                else:
                    QtGui.QMessageBox.about(self, "Error!", "Invalid Texture Percentage Value !")
                    return 0
    
                TEXCount = TEXCount + 1
                   
            EVOLVE.Frac = np.hstack(PerctList) / 100.0
            EVOLVE.EUL = np.dstack(EulList).T
            EVOLVE.TEXCount = TEXCount
                
        self.EVOLVE = EVOLVE
    @pyqtSlot()    
    def PoleFigure(self):
        sqt = np.sqrt(2)
        index, mytexture, ms = self.selectedTEXTURE()
        if index != []:
            Texture = getattr(self, mytexture)
            Eul = Texture.Eul
            XPx, YPx = self.polefig(Eul, [0, 0, 1])
            XPy, YPy = self.polefig(Eul, [1, 1, 0])
            XPz, YPz = self.polefig(Eul, [1, 1, 1])
            figname = ms + " Pole Figures"
            if plt.fignum_exists(figname):
                plt.close(figname)                
            fig = plt.figure(figname)
            axisx = fig.add_subplot(1, 3, 1)
            axisy = fig.add_subplot(1, 3, 2)
            axisz = fig.add_subplot(1, 3, 3)
            axisx.plot(XPx, YPx, '.')
            axisy.plot(XPy, YPy, '.')
            axisz.plot(XPz, YPz, '.')
            
            axisx.set_xlim(-sqt, +sqt)
            axisx.set_ylim(-sqt, +sqt)
            axisy.set_xlim(-sqt, +sqt)
            axisy.set_ylim(-sqt, +sqt)
            axisz.set_xlim(-sqt, +sqt)
            axisz.set_ylim(-sqt, +sqt)
            axisx.set_aspect("equal")
            axisy.set_aspect("equal")
            axisz.set_aspect("equal")
      
            circle = plt.Circle((0, 0), sqt, color='cyan')
            axisx.add_artist(circle)
            circle = plt.Circle((0, 0), sqt, color='cyan')
            axisy.add_artist(circle)
            circle = plt.Circle((0, 0), sqt, color='cyan')
            axisz.add_artist(circle)
            
            axisx.set_title("[0 0 1]", fontsize=25)
            axisy.set_title("[1 1 0]", fontsize=25)
            axisz.set_title("[1 1 1]", fontsize=25)
            axisx.get_xaxis().set_visible(False)
            axisx.get_yaxis().set_visible(False)
            axisy.get_xaxis().set_visible(False)
            axisy.get_yaxis().set_visible(False)
            axisz.get_xaxis().set_visible(False)
            axisz.get_yaxis().set_visible(False)
            
            plt.grid();plt.tight_layout();plt.show()

        
    
    
    def polefig(self, Eul, n):
        Rot1 = conv.zxz_to_rm(Eul)
        Rot = np.transpose(Rot1, (0, 2, 1))
        p = Rot.shape[0]
        N = np.array(n, dtype=np.float64)
        N = N / np.sqrt(N[0] ** 2 + N[1] ** 2 + N[2] ** 2)
        M = np.einsum('kij,j->ki', Rot , N)
        
        theta = np.arccos(M[:, 2])
        phi = np.arctan2(M[:, 1], M[:, 0])
        r = np.sqrt(2 * (1 - np.cos(theta)))
        XXP = r * np.cos(phi)
        YYP = r * np.sin(phi)
        
        XP = []
        YP = []
        for i in range(p):
            if  M[i, 2] >= 0:
                XP.append(XXP[i])
                YP.append(YYP[i])
        XP = np.array(XP)
        YP = np.array(YP)
        return XP, YP

    
    def grabInfoTex(self):
        TEXTURE = TextureFunctions()
        TEXTURE.m = int(self.dimX2DTex.text())
        TEXTURE.n = int(self.dimY2DTex.text())

        if self.checkBox_useseedTexture.isChecked():
            TEXTURE.myseed = int(self.spinBox_myseedTexture.text())
        else:
            TEXTURE.myseed = 0
            
        if self.checkBox_canvas.isChecked():
            TEXTURE.makecanvas = 1
            TEXTURE.p = TEXTURE.m * TEXTURE.n
        else:
            TEXTURE.makecanvas = 0
            TEXTURE.p = int(self.spinBox_pTex.text())
            
        if self.checkBoxKST.isChecked():
            self.KST = 1
        else:
            self.KST = 0

        mycasetexture = "Texture"
        c1 = self.tabWidgetTEXTURE.currentIndex()
        if c1 == 0:
            mycasetexture = mycasetexture + "_Generic"
            
            if self.grainLocCentroid.isChecked():
                TEXTURE.grainloc = 1
            else:
                TEXTURE.grainloc = 0
            
            
            c2 = self.tabWidget_GenericTexture.currentIndex()
            if c2 == 0:
                mycasetexture = mycasetexture + "_Euler"
                mytext = str(self.genTexPhi1.toPlainText())
                TEXTURE.Phi1Func = lambdify('x,y', mytext, self.mydict)
                mytext = str(self.genTexPhi.toPlainText())
                TEXTURE.PhiFunc = lambdify('x,y', mytext, self.mydict)
                mytext = str(self.genTexPhi2.toPlainText())
                TEXTURE.Phi2Func = lambdify('x,y', mytext, self.mydict)
                
            elif c2 == 1:
                mycasetexture = mycasetexture + "_AxisAnglePair"
                mytext = str(self.genTexAxisTheta.toPlainText())
                TEXTURE.AxisThetaFunc = lambdify('x,y', mytext, self.mydict)
                mytext = str(self.genTexAxisPhi.toPlainText())
                TEXTURE.AxisPhiFunc = lambdify('x,y', mytext, self.mydict)
                mytext = str(self.genTexAngle.toPlainText())
                TEXTURE.AngleFunc = lambdify('x,y', mytext, self.mydict)
                
                
                
            elif c2 == 2:
                mycasetexture = mycasetexture + "_RodriguesVector"
                mytext = str(self.genTexRvx.toPlainText())
                TEXTURE.Rvxfunc = lambdify('x,y', mytext, self.mydict)
                mytext = str(self.genTexRvy.toPlainText())
                TEXTURE.Rvyfunc = lambdify('x,y', mytext, self.mydict)
                mytext = str(self.genTexRvz.toPlainText())
                TEXTURE.Rvzfunc = lambdify('x,y', mytext, self.mydict)
                
            
        elif c1 == 1:
            mycasetexture = mycasetexture + "_URO"
            
            if self.radioButton_arvo.isChecked():
                mycasetexture = mycasetexture + '_arvo'
            elif self.radioButton_mack.isChecked():
                mycasetexture = mycasetexture + '_mack'
            elif self.radioButton_miles.isChecked():
                mycasetexture = mycasetexture + '_miles'
            elif self.radioButton_mur.isChecked():
                mycasetexture = mycasetexture + '_mur'
            elif self.radioButton_sm1.isChecked():
                mycasetexture = mycasetexture + '_sm1'
            elif self.radioButton_sm2.isChecked():
                mycasetexture = mycasetexture + '_sm2'
            elif self.radioButton_nxyz.isChecked():
                mycasetexture = mycasetexture + '_nxyz'
            elif self.radioButton_xyz.isChecked():
                mycasetexture = mycasetexture + '_xyz'
            
        elif c1 == 2:
            mycasetexture = mycasetexture + "_Fiber"
            
            TEXTURE.xc = float(self.fibXc.text())
            TEXTURE.yc = float(self.fibYc.text())
            TEXTURE.zc = float(self.fibZc.text())
            TEXTURE.xs = float(self.fibXs.text())
            TEXTURE.ys = float(self.fibYs.text())
            TEXTURE.zs = float(self.fibZs.text())
            TEXTURE.thetamin = float(self.doublespin_thetamin.text()) * pi / 180
            TEXTURE.thetamax = float(self.doublespin_thetamax.text()) * pi / 180
            mytext = str(self.textEdit_fibaxis_dist.toPlainText())
            TEXTURE.axisdist = lambdify('omega', mytext, self.mydict)
            
            
        elif c1 == 3:
            mycasetexture = mycasetexture + "_Sheet"
            TEXTURE.eul1 = float(self.sheeteul1.text()) * pi / 180
            TEXTURE.eul = float(self.sheeteul.text()) * pi / 180
            TEXTURE.eul2 = float(self.sheeteul2.text()) * pi / 180
            
            mytext = str(self.textEdit_gammadist.toPlainText())
            TEXTURE.gammadist = lambdify('gamma', mytext, self.mydict)
            TEXTURE.gammamin = float(self.doubleSpinBox_gammamin.text()) * pi / 180
            TEXTURE.gammamax = float(self.doubleSpinBox_gammamax.text()) * pi / 180
            
            mytext = str(self.textEdit_thetadist.toPlainText())
            TEXTURE.thetadist = lambdify('theta', mytext, self.mydict)
            TEXTURE.thetamin = float(self.doubleSpinBox_thetamin.text()) * pi / 180
            TEXTURE.thetamax = float(self.doubleSpinBox_thetamax.text()) * pi / 180
            
            mytext = str(self.textEdit_phidist.toPlainText())
            TEXTURE.phidist = lambdify('phi', mytext, self.mydict)
            TEXTURE.phimin = float(self.doubleSpinBox_phimin.text()) * pi / 180
            TEXTURE.phimax = float(self.doubleSpinBox_phimax.text()) * pi / 180

            
        TEXTURE.mycasetexture = mycasetexture
        self.TEXTURE = TEXTURE
        self.mycasetexture = mycasetexture
         
       
        
    def runmodeTexture(self):
        self.setEnabled(False)
        self.lcd.display(0)
    
    def closeEvent(self, event):
        
        quit_msg = "Are you sure you want to exit the program?"
        reply = QtGui.QMessageBox.question(self, 'Message',
                         quit_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
    
        if reply == QtGui.QMessageBox.Yes:
            event.accept()
            plt.close("all")
            cv2.destroyAllWindows()
        else:
            event.ignore()
    
    def fillMSList(self):
        self.comboBox_MSList.clear()
        mslist = []
        for i in range(0, 1 + self.EVOLVEcount):
            mystring = 'EVOLVE' + str(i)
            if hasattr(self, mystring) == 1:
                mslist.append("MS" + str(i))
                
        self.comboBox_MSList.addItems(mslist)
            
        
    
    def updateMSView(self):
        self.tvMS.model().clear()
        self.tvMS.model().setHorizontalHeaderLabels(['ID', 'Values'])
        if self.KSM == 1:
            for i in range(0, 1 + self.EVOLVEcount):
                mystring = 'EVOLVE' + str(i)
                if hasattr(self, mystring) == 1:
                    A = getattr(self, mystring)
                    data = {}
                    name = 'MS' + str(i) 
                    data[name] = {}
                    data[name]['Nuclei'] = A.p
                    data[name]['Resolution'] = str(A.m) + 'X' + str(A.n)
    
                           
                    for x in data:
                        if not data[x]:
                            continue
                        parent = QtGui.QStandardItem(x)
                        # parent.setFlags(QtCore.Qt.NoItemFlags)
                        for y in data[x]:
                            value = data[x][y]
                            child0 = QtGui.QStandardItem(y)
                            child0.setFlags(QtCore.Qt.NoItemFlags | 
                                             QtCore.Qt.ItemIsEnabled)
                            child1 = QtGui.QStandardItem(str(value))
                            child1.setFlags(QtCore.Qt.ItemIsEnabled | 
                                             QtCore.Qt.ItemIsEditable | 
                                             ~ QtCore.Qt.ItemIsSelectable)
                            parent.appendRow([child0, child1])
                        self.tvMS.model().appendRow(parent)

                    
                        
        else:
            if hasattr(self, "EVOLVE0"):
                A = self.EVOLVE0
                data = {}
                name = 'MS0' 
                data[name] = {}
                data[name]['Nuclei'] = A.p
                data[name]['Resolution'] = str(A.m) + 'X' + str(A.n)
    
                       
                for x in data:
                    if not data[x]:
                        continue
                    parent = QtGui.QStandardItem(x)
                    # parent.setFlags(QtCore.Qt.NoItemFlags)
                    for y in data[x]:
                        value = data[x][y]
                        child0 = QtGui.QStandardItem(y)
                        child0.setFlags(QtCore.Qt.NoItemFlags | 
                                         QtCore.Qt.ItemIsEnabled)
                        child1 = QtGui.QStandardItem(str(value))
                        child1.setFlags(QtCore.Qt.ItemIsEnabled | 
                                         QtCore.Qt.ItemIsEditable | 
                                         ~ QtCore.Qt.ItemIsSelectable)
                        parent.appendRow([child0, child1])
                    self.tvMS.model().appendRow(parent)
        
        self.tvMS.expandAll()
        self.fillMSList()
    
    
    def updateMSWTView(self):
        self.tvMSWT.model().clear()
        self.tvMSWT.model().setHorizontalHeaderLabels(['ID', 'Values'])
        if self.KSMSWT == 1:
            for i in range(0, 1 + self.MSWTcount):
                name = 'MSWT' + str(i)
                if hasattr(self, name) == 1:
                    A = getattr(self, name)
                    data = {}
                    data[name] = {}
                    data[name]['Nuclei'] = A.p
                    data[name]['Resolution'] = str(A.m) + 'X' + str(A.n)
    
                           
                    for x in data:
                        if not data[x]:
                            continue
                        parent = QtGui.QStandardItem(x)
                        # parent.setFlags(QtCore.Qt.NoItemFlags)
                        for y in data[x]:
                            value = data[x][y]
                            child0 = QtGui.QStandardItem(y)
                            child0.setFlags(QtCore.Qt.NoItemFlags | 
                                             QtCore.Qt.ItemIsEnabled)
                            child1 = QtGui.QStandardItem(str(value))
                            child1.setFlags(QtCore.Qt.ItemIsEnabled | 
                                             QtCore.Qt.ItemIsEditable | 
                                             ~ QtCore.Qt.ItemIsSelectable)
                            parent.appendRow([child0, child1])
                        self.tvMSWT.model().appendRow(parent)

                    
                        
        else:
            if hasattr(self, "MSWT0"):
                A = self.MSWT0
                data = {}
                name = 'MSWT0' 
                data[name] = {}
                data[name]['Nuclei'] = A.p
                data[name]['Resolution'] = str(A.m) + 'X' + str(A.n)
    
                       
                for x in data:
                    if not data[x]:
                        continue
                    parent = QtGui.QStandardItem(x)
                    # parent.setFlags(QtCore.Qt.NoItemFlags)
                    for y in data[x]:
                        value = data[x][y]
                        child0 = QtGui.QStandardItem(y)
                        child0.setFlags(QtCore.Qt.NoItemFlags | 
                                         QtCore.Qt.ItemIsEnabled)
                        child1 = QtGui.QStandardItem(str(value))
                        child1.setFlags(QtCore.Qt.ItemIsEnabled | 
                                         QtCore.Qt.ItemIsEditable | 
                                         ~ QtCore.Qt.ItemIsSelectable)
                        parent.appendRow([child0, child1])
                    self.tvMSWT.model().appendRow(parent)
        
        self.tvMSWT.expandAll()
        # self.fillMSWTList()
    
    
    def updateTEXTUREView(self):
        treeview = self.tvTEX
        treeview.model().clear()
        treeview.model().setHorizontalHeaderLabels(['ID', 'Texture'])
        
        for ss in range(4):
            texturekey = list(self.texturedict.values())[ss]
            if self.KST == 1:
                for i in range(0, 1 + self.TEXTUREcount):
                    name = texturekey + str(i) 
                    if hasattr(self, name) == 1:
                        A = getattr(self, name)
                        data = {}
                        data[name] = {}
                        data[name]['p'] = A.p
                        data[name]['Euler'] = A.Eul


                        for x in data:
                            if not data[x]:
                                continue
                            parent = QtGui.QStandardItem(x)
                            # parent.setFlags(QtCore.Qt.NoItemFlags)
                            for y in data[x]:
                                value = data[x][y]
                                child0 = QtGui.QStandardItem(y)
                                child0.setFlags(QtCore.Qt.NoItemFlags | 
                                                 QtCore.Qt.ItemIsEnabled)
                                child1 = QtGui.QStandardItem(str(value))
                                child1.setFlags(QtCore.Qt.ItemIsEnabled | 
                                                 QtCore.Qt.ItemIsEditable | 
                                                 ~ QtCore.Qt.ItemIsSelectable)
                                parent.appendRow([child0, child1])
                            treeview.model().appendRow(parent)
    
            else:
                name = texturekey + str(0) 
                if hasattr(self, name):
                    A = getattr(self, name)
                    data = {}
                    data[name] = {}
                    data[name]['p'] = A.p
                    data[name]['Euler'] = A.Eul

                    for x in data:
                        if not data[x]:
                            continue
                        parent = QtGui.QStandardItem(x)
                        # parent.setFlags(QtCore.Qt.NoItemFlags)
                        for y in data[x]:
                            value = data[x][y]
                            child0 = QtGui.QStandardItem(y)
                            child0.setFlags(QtCore.Qt.NoItemFlags | 
                                             QtCore.Qt.ItemIsEnabled)
                            child1 = QtGui.QStandardItem(str(value))
                            child1.setFlags(QtCore.Qt.ItemIsEnabled | 
                                             QtCore.Qt.ItemIsEditable | 
                                             ~ QtCore.Qt.ItemIsSelectable)
                            parent.appendRow([child0, child1])
                        treeview.model().appendRow(parent)
                
        treeview.expandAll()
    
    
    def updateCANVASView(self):
        treeview = self.tvCANVAS
        treeview.model().clear()
        treeview.model().setHorizontalHeaderLabels(['ID', 'Texture'])
        
        for ss in range(4):
            texturekey = list(self.canvasdict.values())[ss]
            if self.KST == 1:
                for i in range(0, 1 + self.CANVAScount):
                    name = texturekey + str(i) 
                    if hasattr(self, name) == 1:
                        A = getattr(self, name)
                        data = {}
                        data[name] = {}
                        data[name]['resolution'] = str(A.m) + 'X' + str(A.n)
                        data[name]['Euler'] = A.EUL
                           
                               
                        for x in data:
                            if not data[x]:
                                continue
                            parent = QtGui.QStandardItem(x)
                            # parent.setFlags(QtCore.Qt.NoItemFlags)
                            for y in data[x]:
                                value = data[x][y]
                                child0 = QtGui.QStandardItem(y)
                                child0.setFlags(QtCore.Qt.NoItemFlags | 
                                                 QtCore.Qt.ItemIsEnabled)
                                child1 = QtGui.QStandardItem(str(value))
                                child1.setFlags(QtCore.Qt.ItemIsEnabled | 
                                                 QtCore.Qt.ItemIsEditable | 
                                                 ~ QtCore.Qt.ItemIsSelectable)
                                parent.appendRow([child0, child1])
                            treeview.model().appendRow(parent)
    
            else:
                name = texturekey + str(0) 
                if hasattr(self, name):
                    A = getattr(self, name)
                    data = {}
                    data[name] = {}
                    data[name]['resolution'] = str(A.m) + 'X' + str(A.n)
                    data[name]['Euler'] = A.EUL
                       
                    for x in data:
                        if not data[x]:
                            continue
                        parent = QtGui.QStandardItem(x)
                        # parent.setFlags(QtCore.Qt.NoItemFlags)
                        for y in data[x]:
                            value = data[x][y]
                            child0 = QtGui.QStandardItem(y)
                            child0.setFlags(QtCore.Qt.NoItemFlags | 
                                             QtCore.Qt.ItemIsEnabled)
                            child1 = QtGui.QStandardItem(str(value))
                            child1.setFlags(QtCore.Qt.ItemIsEnabled | 
                                             QtCore.Qt.ItemIsEditable | 
                                             ~ QtCore.Qt.ItemIsSelectable)
                            parent.appendRow([child0, child1])
                        treeview.model().appendRow(parent)
                
        treeview.expandAll()
    
        
    def donemodeTexture(self):
        self.lcd2.display(self.TEXTURE.exetime)
        state = self.tabWidgetTEXTURE.currentIndex()
        if self.KST == 1:
            if self.TEXTURE.makecanvas == 0:            
                self.TEXTUREcount = self.TEXTUREcount + 1
                texturename = list(self.texturedict.values())[state] + str(self.TEXTUREcount)
                setattr(self, texturename, self.TEXTURE)
                self.updateTEXTUREView()
            else:
                self.CANVAScount = self.CANVAScount + 1 
                texturename = list(self.canvasdict.values())[state] + str(self.CANVAScount)
                setattr(self, texturename, self.TEXTURE)
                self.updateCANVASView()  
            
        else:
            if self.TEXTURE.makecanvas == 0:            
                self.ClearAllTex()
                self.TEXTUREcount = 0
                texturename = list(self.texturedict.values())[state] + str(self.TEXTUREcount)
                setattr(self, texturename, self.TEXTURE)
                self.updateTEXTUREView()
                
            else:
                self.ClearAllTexCANVAS()
                self.CANVAScount = 0
                texturename = list(self.canvasdict.values())[state] + str(self.CANVAScount)
                setattr(self, texturename, self.TEXTURE)
                self.updateCANVASView() 
                
        delattr(self, 'TEXTURE')
        self.setEnabled(True)

        
    def Evolve(self):
        self.setEnabled(False)
        self.lcd.display(0)       
        self.grabInfoMS()
        print (self.mycase)        
        getattr(self.EVOLVE, self.mycase)(self.EVOLVE)
        if self.EVOLVE.sa == 0:
            cv2.namedWindow("Microstructure Evolution Random Color", cv2.WINDOW_NORMAL)
            cv2.imshow("Microstructure Evolution Random Color", self.EVOLVE.Col) 
            
        self.lcd.display(self.EVOLVE.exetime)
        if self.KSM == 1:
            self.EVOLVEcount = self.EVOLVEcount + 1
               
        else:
            self.EVOLVEcount = 0
            self.ClearAllMS()
                    
        mystring = 'EVOLVE' + str(self.EVOLVEcount)
        setattr(self, mystring, self.EVOLVE)
        delattr(self, 'EVOLVE')
        self.updateMSView()
        
        self.setEnabled(True)
        
    def GenerateTexture(self):
        self.runmodeTexture()
        self.grabInfoTex()
        getattr(self.TEXTURE, self.mycasetexture)(self.TEXTURE)
        self.donemodeTexture()    

            
    def ChooseTexture(self):
        self.next()
        
    def ShowMicrostructure(self):
        index, myevolve, ms = self.selectedEVOLVE()
        A = getattr(self, myevolve)
        p = A.p
        I = A.I
        Eul = self.TextureURT.Eul
        C = col.EulToCol(I, Eul, p)
        cv2.namedWindow("Microstructure Evolution Random Color", cv2.WINDOW_NORMAL)
        cv2.imshow("Microstructure Evolution Random Color", C)

    
    def Dumptoipy(self):
        self.ipyConsole.pushVariables(dict(self=self))
    
    def Clearipy(self):
        self.ipyConsole.executeCommand("if 'self' in globals(): del self")

    
    def grid(self, x, y, z, resX, resY):
        "Convert 3 column data to matplotlib grid"
        xi = np.linspace(min(x), max(x), resX)
        yi = np.linspace(min(y), max(y), resY)
        Z = griddata(x, y, z, xi, yi, interp='linear')
        X, Y = np.meshgrid(xi, yi)
        return X, Y, Z

class TextureFunctions():
    def Texture_URO_arvo(self, obj):
        obj = tx.Texture_URO_arvo(obj)
        return obj
    
    def Texture_URO_mack(self, obj):
        obj = tx.Texture_URO_mack(obj)
        return obj
    
    def Texture_URO_miles(self, obj):
        obj = tx.Texture_URO_miles(obj)
        return obj
    
    def Texture_URO_mur(self, obj):
        obj = tx.Texture_URO_mur(obj)
        return obj
    
    def Texture_URO_sm1(self, obj):
        obj = tx.Texture_URO_sm1(obj)
        return obj
    
    def Texture_URO_sm2(self, obj):
        obj = tx.Texture_URO_sm2(obj)
        return obj
    
    def Texture_URO_nxyz(self, obj):
        obj = tx.Texture_URO_nxyz(obj)
        return obj 
    
    def Texture_URO_xyz(self, obj):
        obj = tx.Texture_URO_xyz(obj)
        return obj
    
    def Texture_Fiber(self, obj):
        obj = tx.Texture_Fiber(obj)    
        return obj
    
    def Texture_Sheet(self, obj):
        obj = tx.Texture_Sheet(obj)

    
    
    def Texture_Generic_Euler(self, obj):
        pass
    
    def Texture_Generic_AxisAnglePair(self, obj):
        pass
    
    def Texture_Generic_RodriguesVector(self, obj):
        pass

class MSWithTexture():
    def Assign_AS_RELATIVE_FRACTION(self, obj):
        obj = assign.Assign_AS_RELATIVE_FRACTION(obj)
        return obj
        
        
    
    


class EvolutionFunctions():
        
    
    def Evolve_2D_Isotropic_SiteSaturated(self, obj):
        if obj.gr2D == []:    
            obj = ev.Evolve_2D_Isotropic_SiteSaturated_without_gr2D(obj)
        else:
            obj = ev.Evolve_2D_Isotropic_SiteSaturated_with_gr2D(obj)
        return obj
    
    def Evolve_2D_Isotropic_Continuous(self, obj):
        if obj.gr2D == []:    
            obj = ev.Evolve_2D_Isotropic_Continuous_without_gr2D(obj)
        else:
            obj = ev.Evolve_2D_Isotropic_Continuous_with_gr2D(obj)
        return obj
    
    def Evolve_2D_Anisotropic_SiteSaturated_NeighbourHoodBased(self, obj):
        if obj.neightype == 4 and obj.gr2D == []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_NeighbourHoodBased_N4_without_gr2D(obj)
        elif obj.neightype == 4 and obj.gr2D != []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_NeighbourHoodBased_N4_with_gr2D(obj)
        elif obj.neightype == 6 and obj.gr2D == []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_NeighbourHoodBased_N6_without_gr2D(obj)
        elif obj.neightype == 6 and obj.gr2D != []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_NeighbourHoodBased_N6_with_gr2D(obj)
        elif obj.neightype == 8 and obj.gr2D == []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_NeighbourHoodBased_N8_without_gr2D(obj)
        elif obj.neightype == 8 and obj.gr2D != []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_NeighbourHoodBased_N8_with_gr2D(obj)
        return obj
    
    def Evolve_2D_Anisotropic_Continuous_NeighbourHoodBased(self, obj):
        if obj.neightype == 4 and obj.gr2D == []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_NeighbourHoodBased_N4_without_gr2D(obj)
        elif obj.neightype == 4 and obj.gr2D != []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_NeighbourHoodBased_N4_with_gr2D(obj)
        elif obj.neightype == 6 and obj.gr2D == []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_NeighbourHoodBased_N6_without_gr2D(obj)
        elif obj.neightype == 6 and obj.gr2D != []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_NeighbourHoodBased_N6_with_gr2D(obj)
        elif obj.neightype == 8 and obj.gr2D == []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_NeighbourHoodBased_N8_without_gr2D(obj)
        elif obj.neightype == 8 and obj.gr2D != []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_NeighbourHoodBased_N8_with_gr2D(obj)
        return obj
    
    def Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical(self, obj):
        if obj.Rfunc == [] and obj.Thetafunc == [] and obj.Adotfunc == []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_without_aspect_without_theta_without_adot(obj)
        elif obj.Rfunc != [] and obj.Thetafunc == [] and obj.Adotfunc == []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_with_aspect_without_theta_without_adot(obj)
        elif obj.Rfunc == [] and obj.Thetafunc != [] and obj.Adotfunc == []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_without_aspect_with_theta_without_adot(obj)
        elif obj.Rfunc == [] and obj.Thetafunc == [] and obj.Adotfunc != []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_without_aspect_without_theta_with_adot(obj)
        elif obj.Rfunc != [] and obj.Thetafunc != [] and obj.Adotfunc == []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_with_aspect_with_theta_without_adot(obj)
        elif obj.Rfunc == [] and obj.Thetafunc != [] and obj.Adotfunc != []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_without_aspect_with_theta_with_adot(obj)
        elif obj.Rfunc != [] and obj.Thetafunc == [] and obj.Adotfunc != []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_with_aspect_without_theta_with_adot(obj)
        elif obj.Rfunc != [] and obj.Thetafunc != [] and obj.Adotfunc != []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_Elliptical_with_aspect_with_theta_with_adot(obj)
        return obj
    def Evolve_2D_Anisotropic_Continuous_Generic_Elliptical(self, obj):
        # print (vars(obj))
        if obj.Rfunc == [] and obj.Thetafunc == [] and obj.Adotfunc == []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_without_aspect_without_theta_without_adot(obj)
        elif obj.Rfunc != [] and obj.Thetafunc == [] and obj.Adotfunc == []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_with_aspect_without_theta_without_adot(obj)
        elif obj.Rfunc == [] and obj.Thetafunc != [] and obj.Adotfunc == []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_without_aspect_with_theta_without_adot(obj)
        elif obj.Rfunc == [] and obj.Thetafunc == [] and obj.Adotfunc != []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_without_aspect_without_theta_with_adot(obj)
        elif obj.Rfunc != [] and obj.Thetafunc != [] and obj.Adotfunc == []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_with_aspect_with_theta_without_adot(obj)
        elif obj.Rfunc == [] and obj.Thetafunc != [] and obj.Adotfunc != []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_without_aspect_with_theta_with_adot(obj)
        elif obj.Rfunc != [] and obj.Thetafunc == [] and obj.Adotfunc != []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_with_aspect_without_theta_with_adot(obj)
        elif obj.Rfunc != [] and obj.Thetafunc != [] and obj.Adotfunc != []:
            obj = ev.Evolve_2D_Anisotropic_Continuous_Generic_Elliptical_with_aspect_with_theta_with_adot(obj)
        return obj
    def Evolve_2D_Anisotropic_SiteSaturated_Generic_DirectionBased(self, obj):
        if obj.Grvector == [] and obj.Thetafunc == []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_DirectionBased_without_theta_without_gr(obj)
        elif obj.Grvector != [] and obj.Thetafunc == []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_DirectionBased_without_theta_with_gr(obj)
        elif obj.Grvector == [] and obj.Thetafunc != []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_DirectionBased_with_theta_without_gr(obj)
        elif obj.Grvector != [] and obj.Thetafunc != []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_DirectionBased_with_theta_with_gr(obj)
            
    def Evolve_2D_Anisotropic_SiteSaturated_Generic_TextureBased_Ellipsoidal(self, obj):
        if obj.Rabfunc == [] and obj.Racfunc == [] and obj.Adotfunc == []:
            obj = ev.Evolve_2D_Anisotropic_SiteSaturated_Generic_TextureBased_Ellipsoidal_without_ab_without_ac_without_adot(obj)
        elif obj.Rabfunc != [] and obj.Racfunc == [] and obj.Adotfunc == []:
            print("kaka")
        elif obj.Rabfunc == [] and obj.Racfunc != [] and obj.Adotfunc == []:
            print("kaka")
        elif obj.Rabfunc == [] and obj.Racfunc == [] and obj.Adotfunc != []:
            print("kaka")
        elif obj.Rabfunc != [] and obj.Racfunc != [] and obj.Adotfunc == []:
            print("kaka")
        elif obj.Rabfunc == [] and obj.Racfunc != [] and obj.Adotfunc != []:
            print("kaka")
        elif obj.Rabfunc != [] and obj.Racfunc == [] and obj.Adotfunc != []:
            print("kaka")
        elif obj.Rabfunc != [] and obj.Racfunc != [] and obj.Adotfunc != []:
            print("kaka")

class SetPlottingOptions(PlottingOptionsBase, PlottingOptionsUI):                                
    
    def __init__(self, obj, parent=None):
        
        self.pops = obj.pops
        PlottingOptionsBase.__init__(self, parent)
        self.setupUi(self)
        
        
    def SetOptions(self):
        self.pops.colmaps = str(self.mptcolmaps.currentText())
        return self.pops
        
        

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = EvoSimGui()
    window.show()
    '''
    h = app.desktop().screenGeometry().height()
    w = app.desktop().screenGeometry().width()
    window.dimX2D.setValue(h)
    window.dimY2D.setValue(w)
    '''
    window.setStyle(QtGui.QStyleFactory.create('cleanlooks'))
    sys.exit(app.exec_())
