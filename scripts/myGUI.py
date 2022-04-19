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

class myGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Python Project Build v1.0 [Personal Repo]')
        self.setGeometry(300, 300, 500, 500) # mixture of move(x, y) and resize(width, height)

        # Initialize tab widget
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)

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

        # add filemenu [TOP: File] --> imageMenu [Middle: 'Import'] --> datasetMenu [Bottom: 'Import mail]
        import_subMenu = QMenu('Import', self)
        imageMenu = QAction('Image Files', self)
        imageMenu.triggered.connect(self.image_files_window)
        datasetMenu = QAction('Datasets', self)
        import_subMenu.addAction(imageMenu)
        import_subMenu.addAction(datasetMenu)
        filemenu.addMenu(import_subMenu)

        # sub menu to open/close tabs in our window
        # tab 1
        new_tab_menu = QAction('Open New Tab 1', self)
        filemenu.addAction(new_tab_menu)
        new_tab_menu.triggered.connect(self.add_tab_1)  # link (signal) to open tab function
        
        # tab 2
        new_tab_menu2 = QAction('Open New Tab 2', self)
        filemenu.addAction(new_tab_menu2)
        new_tab_menu2.triggered.connect(self.add_tab_2)

        # tab 3
        new_tab_menu3 = QAction('Open New Tab 3', self)
        filemenu.addAction(new_tab_menu3)
        new_tab_menu3.triggered.connect(self.add_tab_3)

        # close current tab
        remove_tab_menu = QAction('Close Current Tab', self)
        filemenu.addAction(remove_tab_menu)
        remove_tab_menu.triggered.connect(self.removeTab)

        # close all open tabs
        remove_all_tab_menu = QAction('Close All Tabs', self)
        filemenu.addAction(remove_all_tab_menu)
        remove_all_tab_menu.triggered.connect(self.removeAllTabs)

        # Add view menu
        viewmenu = menubar.addMenu('&View')
        view_data = QAction('View data', self)
        viewmenu.addAction(view_data)
        view_data.triggered.connect(self.view_data_window)  # launch new 'View Data' window

        version_history = QMenu('Version history', self)
        version_number = QAction('Version number', self)
        version_history.addAction(version_number)
        viewmenu.addMenu(version_history)

        # set a status bar at bottom of window
        self.statusBar().showMessage('Ready')
        
        # ====== Add toolbars ======
        fileToolBar = self.addToolBar('Tab control')    # set name of toolbar
        fileToolBar.addAction(remove_all_tab_menu)
        fileToolBar.addAction(remove_tab_menu)
        
        otherToolBar = self.addToolBar('Other control')
        otherToolBar.addAction(imageMenu) # opens new window to view images

        self.show()  # make visible

        
    # signal to connect view_data signal
    # every time a QAction is trigger signal emitted, this signal must be connectedto some function, this can be done using slot function
    @pyqtSlot()
    # open new windows
    def view_data_window(self):
        new_window = QDialog(self)
        new_window.setWindowTitle('View Data')
        new_window.resize(300, 400)
        new_window.exec_()

    def image_files_window(self):
        new_window2 = QDialog(self)
        new_window2.setWindowTitle('Image Files')
        new_window2.resize(300, 400)
        new_window2.exec_()

    # open/close tabs
    def add_tab_1(self):
        self.table_widget.tabs.addTab(tab_1_widget(), "Tab 1")

    def add_tab_2(self):
        self.table_widget.tabs.addTab(drawCanvas(), "Tab 2")

    def add_tab_3(self):
        self.table_widget.tabs.addTab(QWidget(), "Tab 3")

    def removeTab(self):
        current_tab_index = self.table_widget.tabs.currentIndex()
        self.table_widget.tabs.removeTab(current_tab_index)

    # function to remove/close all tabs
    def removeAllTabs(self):
        tab_count = self.table_widget.tabs.count()  # returns how many tabs are open
        while(tab_count != 0):
            self.table_widget.tabs.removeTab(tab_count)
            tab_count = tab_count - 1
        self.table_widget.tabs.removeTab(0)


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
        self.tabs.setLayout(self.tabs.layout)
        self.layout.addWidget(self.tabs)    # Add tabs to widget

        # Exit button to close tabs
        #tabExitButton = QPushButton("Exit", self)
        #tabExitButton.move(15,25)
        #self.tabs.layout.addWidget(tabExitButton)

class tab_1_widget(QWidget):
    def __init__(self, parent=None):
        super(tab_1_widget, self).__init__(parent)
        self.main_layout = QVBoxLayout(self)
        text_label = QLabel("This is Tab 1")
        self.main_layout.addWidget(text_label)

class drawCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(drawCanvas, self).__init__(parent)

        # set up layout 
        self.canvas_layout = QVBoxLayout(self)
        self.canvas_label = QLabel()
        
        # add text label
        text_label = QLabel("This is Tab 2: Drawing Canvas")
        self.canvas_layout.addWidget(text_label)

        # set up canvas
        self.canvas = QtGui.QPixmap(400, 400)
        self.canvas.fill(QtGui.QColor('white'))
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
        # First event
        if self.last_x is None: 
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time

        self.painter = QtGui.QPainter(self.canvas_label.pixmap())

        # set thickness of pen
        p = self.painter.pen()
        p.setWidth(5)
        self.painter.setPen(p)

        self.painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        self.painter.end()
        self.update()

        # Update the origin for next time
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

        # save drawn image
        img = QtGui.QPixmap(self.canvas_label.pixmap())
        img.save("images\loadedimage.png")


