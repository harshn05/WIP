import os, sys

from PyQt4 import QtCore, QtGui


home = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
sys.path.append(os.path.join(home, 'site-packages'))
app = QtGui.QApplication(sys.argv)
class Window(QtGui.QTableView):
    def __init__(self):
        super(Window, self).__init__() 
        model = QtGui.QStandardItemModel()

        model.setHorizontalHeaderLabels(['Column 0', 'Column 1', 'Column 3'])
        for i in range(3):
            item = QtGui.QStandardItem('Column 0 Item %s' % i)
            item.setData('< Column 0 Custom Value as "UserRole" >', QtCore.Qt.UserRole)
            item.setData('< Column 0 Custom Value as "UserRole+1" >', QtCore.Qt.UserRole + 1)
            model.appendRow(item)

            itemRow = item.row()
            indexOfColumn1 = model.index(itemRow, 1)
            indexOfColumn2 = model.index(itemRow, 2)

            model.setData(indexOfColumn1, 'Column 1 Item', QtCore.Qt.DisplayRole)
            model.setData(indexOfColumn1, '< Column 1 Custom Value as "UserRole" >', QtCore.Qt.UserRole)

            model.setData(indexOfColumn2, 'Column 2 Item', QtCore.Qt.DisplayRole)
            model.setData(indexOfColumn2, '< Column 2 Custom Value as "UserRole" >', QtCore.Qt.UserRole)

        self.setModel(model)
        self.clicked.connect(self.onClick)  
        self.show()

    def onClick(self, index=None):
        row = index.row()
        column = index.column()

        model = self.model()
        indexOfColumn0 = model.index(row, 0)
        indexOfColumn1 = model.index(row, 1)
        indexOfColumn2 = model.index(row, 2)
        print indexOfColumn0 == index, indexOfColumn1 == index, indexOfColumn2 == index


        print  'ROW: %s  COLUMN: %s' % (row, column)
        print 'DispayRoleData via self.model().data(index,role): "%s"' % self.model().data(index, QtCore.Qt.DisplayRole).toString()
        print 'UserRoleData via self.model().data(index,role): "%s"' % self.model().data(index, QtCore.Qt.UserRole).toPyObject()
        print 'UserRoleData via self.model().data(index,role): "%s"' % self.model().data(index, QtCore.Qt.UserRole + 1).toPyObject()      

        for key in self.model().itemData(index):
            print 'self.model.itemData(index):    key: %s  value: %s' % (key, self.model().itemData(index)[key].toString())

        item = self.model().itemFromIndex(index)


window = Window()
sys.exit(app.exec_())
