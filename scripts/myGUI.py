import sys
from PyQt5.QtWidgets import (QMenu, QDialog, QApplication, QTabWidget, QDesktopWidget, QLabel, QWidget, QPushButton, QToolTip, QMainWindow, qApp, QAction, QGridLayout,
                    QHBoxLayout, QVBoxLayout  )
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import (QIcon, QFont)
from PyQt5.Qt import QWidget, QMainWindow
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt

class myGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
        self.show()

    def initUI(self):

        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)

        self.setWindowTitle('Python Project Build v1.0')
        self.setGeometry(400, 400, 600, 600) # mixture of move(x, y) and resize(width, height)

        # set font & size of tool tip
        QToolTip.setFont(QFont('SansSerif', 10))

        # ===== menu bar settings =====
        # add menu bar with icon, shortcut and message (Requires QMainWindow)
        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)
        
        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File') # &File is simpler version of setShortcut() -> 'Alt + F'
        filemenu.addAction(exitAction)

        # sub menu to open to new tab
        new_tab_menu = QAction('Open New Tab 1', self)
        filemenu.addAction(new_tab_menu)
        # link to open tab function
        new_tab_menu.triggered.connect(self.add_tab_1)
        
        # open another tab 
        new_tab_menu2 = QAction('Open New Tab 2', self)
        filemenu.addAction(new_tab_menu2)
        new_tab_menu2.triggered.connect(self.add_tab_2)

        # close tab through menu selection
        remove_tab_menu = QAction('Close Current Tab', self)
        filemenu.addAction(remove_tab_menu)
        remove_tab_menu.triggered.connect(self.removeTab)

        # close all tabs currently open
        remove_all_tab_menu = QAction('Close All Tabs', self)
        filemenu.addAction(remove_all_tab_menu)
        remove_all_tab_menu.triggered.connect(self.removeAllTabs)


        # add filemenu [TOP: File] --> imageMenu [Middle: 'Import'] --> datasetMenu [Bottom: 'Import mail]
        import_subMenu = QMenu('Import', self)
        imageMenu = QAction('Image Files', self)
        datasetMenu = QAction('Datasets', self)
        import_subMenu.addAction(imageMenu)
        import_subMenu.addAction(datasetMenu)
        filemenu.addMenu(import_subMenu)

        # new menu 'view'
        viewmenu = menubar.addMenu('&View')
        view_data = QAction('View data', self)

        viewmenu.addAction(view_data)
        nest_view = QMenu('Version history', self)
        nest_version = QAction('Version number', self)
        nest_view.addAction(nest_version)
        viewmenu.addMenu(nest_view)

        # set a status bar at bottom of window
        self.statusBar().showMessage('Ready')
        
        # ====== Add toolbars ======
        fileToolBar = self.addToolBar('Tab control')    # set name of toolbar
        fileToolBar.addAction(remove_all_tab_menu)
        
        otherToolBar = self.addToolBar('Other control')
        otherToolBar.addAction(imageMenu) # opens new window to view images





        self.show()  # make visible

        view_data.triggered.connect(self.view_data_window)
        imageMenu.triggered.connect(self.image_files_window)

    
    # signal to connect view_data signal
    # every time a QAction is trigger signal emitted, this signal must be connected
    # to some function, this can be done using slot function
    @QtCore.pyqtSlot()
    def add_tab_1(self):
        self.table_widget.tabs.addTab(tab_1_widget(),"Tab 1")
    
    def add_tab_2(self):
        self.table_widget.tabs.addTab(drawCanvas(),"Tab 2")

    def removeTab(self):
        current_tab_index = self.table_widget.tabs.currentIndex() # return current index of open tab
        self.table_widget.tabs.removeTab(current_tab_index)

    # function to remove/close all tabs
    def removeAllTabs(self):
        tab_count = self.table_widget.tabs.count()  # returns how many tabs are open
        while(tab_count != 0):
            self.table_widget.tabs.removeTab(tab_count)
            tab_count = tab_count - 1
        self.table_widget.tabs.removeTab(0)

   

    # open new window
    def view_data_window(self):
        w = QDialog(self)
        w.setWindowTitle('View Data')
        w.resize(640, 480)
        w.exec_()

    # open new window
    def image_files_window(self):
        w1 = QDialog(self)
        w1.setWindowTitle('Image files')
        w1.resize(500, 700)
        w1.exec()

    # center at application launch
    def center(self):
        qr = self.frameGeometry()   # get window information
        cp = QDesktopWidget().availableGeometry().center()  # get monitor information, get center position of monitor
        qr.moveCenter(cp) # set position of monitor to variable 'qr'
        self.move(qr.topLeft()) # move window to center position of monitor








class MyTableWidget(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tabs.resize(300,200)
        self.tabs.layout = QVBoxLayout(self)

        # Exit button to close tabs
        #tabExitButton = QPushButton("Exit", self)
        #tabExitButton.move(15,25)
        #self.tabs.layout.addWidget(tabExitButton)
        self.tabs.setLayout(self.tabs.layout)
        #tabExitButton.clicked.connect(self.tabs.removeTab)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        #self.setLayout(self.layout)

    

class tab_1_widget(QWidget):
    def __init__(self, parent=None):
        super(tab_1_widget, self).__init__(parent)
        self.main_layout = QVBoxLayout(self)
        text_label = QLabel("This is Tab 1")
        self.main_layout.addWidget(text_label)


class drawCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(drawCanvas, self).__init__(parent)

        # set up canvas to draw
        self.canvas_layout = QVBoxLayout(self)
        self.canvas_label = QLabel()
        self.canvas = QtGui.QPixmap(300, 300)
        self.canvas.fill(QtGui.QColor("white"))
        self.canvas_label.setPixmap(self.canvas)
        self.canvas_layout.addWidget(self.canvas_label)
        self.last_x, self.last_y = None, None

        # add button to clear canvas
        clearButton = QPushButton('Clear')
        self.canvas_layout.addWidget(clearButton)

        clearButton.clicked.connect(self.clear_click)


    @QtCore.pyqtSlot()
    # clear canvas
    def clear_click(self):
        self.canvas_label.setPixmap(self.canvas)
    
    # Adapted from https://www.pythonguis.com/tutorials/bitmap-graphics/
    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        self.painter = QtGui.QPainter(self.canvas_label.pixmap())
        self.painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        self.painter.end()
        self.update()


        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None


