import os

from PyQt4.QtCore import QModelIndex, Qt
from PyQt4.QtGui import QApplication, QItemSelectionModel, \
                        QPushButton, QStandardItem, \
                        QStandardItemModel, QTreeView
from PyQt4.uic import loadUi


class PrvTreeviewNest(QTreeView):
    def __init__(self):
        super(PrvTreeviewNest, self).__init__()

        loadUi('treeview_nest.ui')

        # row can be 0 even when it's more than 0.
        self._datamodel = QStandardItemModel(0, 2)
        self.setModel(self._datamodel)

        for i in range(4):
            self.add_widget(i + 1)

        self.show()

    def add_widget(self, n):
        std_item = QStandardItem('{}th item'.format(n))
        self._datamodel.setItem(n, 0, std_item)

        node_widget = QPushButton('{}th button'.format(n))
        qindex_widget = self._datamodel.index(n, 1, QModelIndex())
        self.setIndexWidget(qindex_widget, node_widget)

        if n == 2:
            std_item_child = QStandardItem('child')
            std_item.appendRow(std_item_child)

            node_widget_child = QPushButton('petit button')
            qindex_widget_child = self._datamodel.index(n, 1, QModelIndex())
            self.setIndexWidget(qindex_widget_child, node_widget_child)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    # window = TreeviewWidgetSelectProve()
    window = PrvTreeviewNest()
    # window = TreeviewWidgetSelectProve()
    window.resize(320, 240)
    # window.show();
    window.setWindowTitle(
         QApplication.translate("toplevel", "Top-level widget"))
    # window.add_cols()

    sys.exit(app.exec_())
