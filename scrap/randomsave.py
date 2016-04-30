from __future__ import division

import datetime
import importlib
import platform
import platform
import platform
import signal
import subprocess
import sys
import sys
import sys
import time

from PyQt4 import QtGui, uic, QtCore
import cv2
from sympy import lambdify

import numpy as np
import resourceLIST_rc


# import pyximport; pyximport.install()
class QtCapture(QtGui.QWidget):
    def __init__(self, *args):
        super(QtGui.QWidget, self).__init__()

        self.fps = 40
        self.cap = cv2.VideoCapture(args[0])
        # self.video_frame = QtGui.QLabel()
        self.video_frame = args[1]
        # #lay = QtGui.QVBoxLayout()
        # lay.setMargin(0)
        # lay.addWidget(self.video_frame)
        # self.setLayout(lay)

    def setFPS(self, fps):
        self.fps = fps

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        # My webcam yields frames in BGR format
        # frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
        frame = np.random.random((400, 400))
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_MonoLSB)
        pix = QtGui.QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000. / self.fps)

    def stop(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QtGui.QWidget, self).deleteLater()



class ControlWindow(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.capture = None

        self.start_button = QtGui.QPushButton('Start')
        self.start_button.clicked.connect(self.startCapture)
        self.quit_button = QtGui.QPushButton('End')
        self.quit_button.clicked.connect(self.endCapture)
        self.end_button = QtGui.QPushButton('Stop')
        self.video_frame = QtGui.QLabel()

        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.quit_button)
        vbox.addWidget(self.video_frame)
        self.setLayout(vbox)
        self.setWindowTitle('Control Panel')
        self.setGeometry(100, 100, 200, 200)
        self.show()

    def startCapture(self):
        if not self.capture:
            self.capture = QtCapture(0, self.video_frame)
            self.end_button.clicked.connect(self.capture.stop)
            # self.capture.setFPS(1)
            self.capture.setParent(self)
            self.capture.setWindowFlags(QtCore.Qt.Tool)
        self.capture.start()
        self.capture.show()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())
    
