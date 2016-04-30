import sys
import sys

from PyQt4 import QtCore, QtGui, uic 

import PyQt4.QtGui as QtGui
import numpy as np


qtCreatorFile = "./uicomponents/Texturizer.ui"  # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class Texturizer(QtGui.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        
    def setMax(self):
        print (TATA)
        

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = Texturizer()
    window.show()
    sys.exit(app.exec_())
