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
        self.setGeometry(300, 300, 600, 600) # mixture of move(x, y) and resize(width, height)

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
        import_dataset_menu = QAction('Import Datasets', self)
        filemenu.addAction(import_dataset_menu)
        import_dataset_menu.triggered.connect(self.dataset_window)


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

        # tab 4
        new_tab_menu4 = QAction('Open New Tab 4', self)
        filemenu.addAction(new_tab_menu4)
        new_tab_menu4.triggered.connect(self.add_tab_4)

        # tab 5
        new_tab_menu5 = QAction('Open New Tab 5', self)
        filemenu.addAction(new_tab_menu5)
        new_tab_menu5.triggered.connect(self.add_tab_5)

        # close current tab
        remove_tab_menu = QAction('Close Current Tab', self)
        filemenu.addAction(remove_tab_menu)
        remove_tab_menu.triggered.connect(self.removeTab)

        # close all open tabs
        remove_all_tab_menu = QAction('Close All Tabs', self)
        filemenu.addAction(remove_all_tab_menu)
        remove_all_tab_menu.triggered.connect(self.removeAllTabs)

        # Add view menu
        # Add view training images and testing images
        view_menu = menubar.addMenu('&View')
        view_training_images_menu = QAction('View Training Images', self)
        view_menu.addAction(view_training_images_menu)

        view_training_images_menu.triggered.connect(self.view_data_window)

        view_testing_images_menu = QAction('View Testing Images', self)
        view_menu.addAction(view_testing_images_menu)

        version_history = QMenu('Version history', self)
        version_number = QAction('Version number', self)
        version_history.addAction(version_number)
        view_menu.addMenu(version_history)

        # set a status bar at bottom of window
        self.statusBar().showMessage('Ready')
        
        # ====== Add toolbars ======
        fileToolBar = QToolBar(self)
        self.addToolBar(QtCore.Qt.TopToolBarArea, fileToolBar)
        #fileToolBar = self.addToolBar('Tab control')    # set name of toolbar
        fileToolBar.addAction(remove_all_tab_menu)
        fileToolBar.addAction(remove_tab_menu)
        
        #otherToolBar = self.addToolBar('Other control')
        fileToolBar.addAction(import_dataset_menu)

        # Additional toolbar to left side of window
        sideToolBar = QToolBar(self)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, sideToolBar)
        sideToolBar.addAction(import_dataset_menu)
        sideToolBar.addAction(new_tab_menu)
        sideToolBar.addAction(new_tab_menu2)
        sideToolBar.addAction(new_tab_menu3)
        sideToolBar.addAction(new_tab_menu4)
        sideToolBar.addAction(new_tab_menu5)

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

    def dataset_window(self):
        new_window3 = dataset_Dialog_window()
        new_window3.setWindowTitle('Import Dataset')
        new_window3.resize(400, 400)
        new_window3.exec_()

    # functions open/close tabs
    # when tab is opened automatically switches to new opened tab
    def add_tab_1(self):
        # saves current index our main window is open at
        current_index = self.table_widget.tabs.currentIndex()
        # opens new tab at that particular index
        self.table_widget.tabs.insertTab(current_index, tab_1_widget(), "Tab 1")
        # sets current index to the index of newly opened tab
        self.table_widget.tabs.setCurrentIndex(current_index)

    def add_tab_2(self):
        current_index = self.table_widget.tabs.currentIndex()
        self.table_widget.tabs.insertTab(current_index, drawCanvas(), "Tab 2")
        self.table_widget.tabs.setCurrentIndex(current_index)

    def add_tab_3(self):
        current_index = self.table_widget.tabs.currentIndex()
        self.table_widget.tabs.insertTab(current_index, tab_3_widget(), "Tab 3")
        self.table_widget.tabs.setCurrentIndex(current_index)

    def add_tab_4(self):
        current_index = self.table_widget.tabs.currentIndex()
        self.table_widget.tabs.insertTab(current_index, tab_4_widget(), "Tab 4")
        self.table_widget.tabs.setCurrentIndex(current_index)

    def add_tab_5(self):
        current_index = self.table_widget.tabs.currentIndex()
        self.table_widget.tabs.insertTab(current_index, tab_5_widget(), "Tab 5")
        self.table_widget.tabs.setCurrentIndex(current_index)

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



# for dataset download/train window
class dataset_Dialog_window(QDialog):
    def __init__(self):
        super(QDialog, self).__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setGeometry(300, 300, 300, 300)

        self.text = QLabel("This is the dataset window")
        self.layout.addWidget(self.text)

        self.text_box = QTextBrowser(self)
        welcome_message = "Hello"
        self.layout.addWidget(self.text_box)
        self.text_box.setText(welcome_message)

        instruction_text = QLabel("Select a model below")
        self.layout.addWidget(instruction_text)

        comboButton = QComboBox(self)
        self.layout.addWidget(comboButton)
        option_array = ["Select", "Option 1", "Option 2"]
        comboButton.addItems(option_array)

        download_button = QPushButton('Download', self)
        self.layout.addWidget(download_button)

        train_button = QPushButton('Train', self)
        self.layout.addWidget(train_button)

        cancel_button = QPushButton('Cancel')
        self.layout.addWidget(cancel_button)


# main widget to control tabs
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

# Tab 3 will display the View Training Images
class tab_3_widget(QWidget):
    def __init__(self, parent = None):
        super(tab_3_widget, self).__init__(parent)

        layout = QVBoxLayout(self)
        text_label = QLabel("This is Tab 3: View Training Images")
        layout.addWidget(text_label)

# Tab 4 will display the View Testing Images
class tab_4_widget(QWidget):
    def __init__(self, parent = None):
        super(tab_4_widget, self).__init__(parent)

        layout = QVBoxLayout(self)
        text_label = QLabel("This is Tab 4: View Testing Images")
        layout.addWidget(text_label)

# Tab 5 will display the Import Datasets
class tab_5_widget(QWidget):
    def __init__(self, parent = None):
        super(tab_5_widget, self).__init__(parent)

        self.layout = QVBoxLayout(self)
        text_label = QLabel("This is Tab 5: Import Datasets")
        self.layout.addWidget(text_label)

        #self.layout = QVBoxLayout()
        #self.setLayout(self.layout)
        #self.setGeometry(300, 300, 300, 300)

        self.text = QLabel("This is the dataset window")
        self.layout.addWidget(self.text)

        self.text_box = QTextBrowser(self)
        welcome_message = "Hello"
        self.layout.addWidget(self.text_box)
        self.text_box.setText(welcome_message)

        instruction_text = QLabel("Select a model below")
        self.layout.addWidget(instruction_text)

        comboButton = QComboBox(self)
        self.layout.addWidget(comboButton)
        option_array = ["Select", "Option 1", "Option 2"]
        comboButton.addItems(option_array)

        download_button = QPushButton('Download', self)
        self.layout.addWidget(download_button)

        train_button = QPushButton('Train', self)
        self.layout.addWidget(train_button)

        cancel_button = QPushButton('Cancel')
        self.layout.addWidget(cancel_button)

        
# Tab 2
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
        # in event of window resize, the canvas size will adjust automatically - refer to resizeEvent()
        canvas_resize = QtGui.QPixmap(500, 500)
        self.canvas = canvas_resize.scaled(self.width(), self.height())
        self.canvas.fill(QtGui.QColor('white'))
        canvas_resize.fill(QtGui.QColor('white'))
        self.canvas_label.setPixmap(self.canvas)
        self.canvas_label.setMinimumSize(1,1)
        self.canvas_layout.addWidget(self.canvas_label)

        self.last_x, self.last_y = None, None

        text_label = QLabel('Prediction Here: ')
        self.canvas_layout.addWidget(text_label)

        self.text_label_height = 20 # used to offset cursor
        print(self.text_label_height)

        text_box = QTextBrowser(self)
        self.canvas_layout.addWidget(text_box)
        prediction_text = "Prediction is: ####"
        text_box.setText(prediction_text)

        self.text_box_height = text_box.height() # used to offset cursor
        
        # total offset from widgets, added to y-position
        self.total_offset = self.text_label_height + self.text_box_height

        # add button to clear canvas
        clearButton = QPushButton('Clear')
        self.canvas_layout.addWidget(clearButton)
        clearButton.clicked.connect(self.clear_click)


    @QtCore.pyqtSlot()
    # resize event of canvas
    def resizeEvent(self, e):
        canvas_resize = QtGui.QPixmap(500, 500)
        canvas_resize.fill(QtGui.QColor('white'))
        self.canvas = canvas_resize.scaled(self.width(), self.height())
        self.canvas_label.setPixmap(self.canvas)
        self.canvas_label.resize(self.width(), self.height())

    # clear canvas
    def clear_click(self):
        self.canvas_label.setPixmap(self.canvas)

    # get mouse/curosr position
    def mousePressEvent(self, e):
        #print(e.pos())
        pass
    
    # Adapted from https://www.pythonguis.com/tutorials/bitmap-graphics/
    def mouseMoveEvent(self, e):
        #cursor = QtGui.QCursor()
        #print(cursor.pos())

        # First event
        if self.last_x is None: 
            self.last_x = e.x()
            self.last_y = e.y() + self.total_offset
            return # Ignore the first time

        self.painter = QtGui.QPainter(self.canvas_label.pixmap())

        # set thickness of pen
        p = self.painter.pen()
        p.setWidth(5)
        self.painter.setPen(p)
        self.painter.drawLine(self.last_x, self.last_y, e.x(), e.y() + self.total_offset)
        self.painter.end()
        self.update()

        # Update the origin for next time
        self.last_x = e.x()
        self.last_y = e.y()+ self.total_offset

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

        # save drawn image
        img = QtGui.QPixmap(self.canvas_label.pixmap())
        img.save("images\loadedimage.png")


