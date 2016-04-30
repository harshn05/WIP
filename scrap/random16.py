from PyQt4 import QtGui, QtCore, uic
from PyQt4.QtCore import pyqtSlot, SIGNAL, SLOT


msmodel = QtGui.QStandardItemModel(0, 2)
data = {}
name = 'MS' + str(1) 
data[name] = {}
data[name]['Nuclei'] = "N"
data[name]['ID'] = "id"
data[name]['Resolution'] = "Res"
data[name]['Grain Structure'] = "GS"
# parent = self.parent

                      
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
        # parent.appendRow([child0, child1])
        msmodel.appendRow([child0, child1])
    msmodel.appendRow(parent)
print msmodel.__dict__
