import sys

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *


data = root = [

    ("Linux", [

        ("System",

                [("System name", []),
         ("Kernel", []),
         ("Drivers", []),
         ("Memory", []),
         ("Processes", []),
                 ("Disk mounted", []),
         ("Services Running", []),
         ("Installed Packages", [])]),
        # [("System name", [])]),

        ("Network",
        [("Nework confi.", []),
        ("Interface test", [])]),

        ("PCI Devices",
        [("PCI devices", [])]),

        ("Logs",
        [("Messages", []),
        ("Dmesg", [])]),


        ])]

class Window(QtGui.QWidget):

    def __init__(self):

        QtGui.QWidget.__init__(self)

        self.treeView = QtGui.QTreeView()


        self.model = QtGui.QStandardItemModel()
        self.addItems(self.model, data)
        self.treeView.setModel(self.model)

        self.model.setHorizontalHeaderLabels([self.tr("Object")])

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.treeView)
        self.setLayout(layout)
        self.treeView.connect(self.treeView, QtCore.SIGNAL('clicked(QModelIndex)'), self.treefunction)

    def treefunction(self, index):
        print(index.model().itemFromIndex(index).text())
        


    def addItems(self, parent, elements):

        for text, children in elements:
            item = QtGui.QStandardItem(text)
            parent.appendRow(item)
            if children:
                self.addItems(item, children)
if __name__ == "__main__":

    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
