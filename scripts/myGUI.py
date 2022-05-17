import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtGui
from PyQt5.QtGui import *

import torch

import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from PyQt5.QtCore import QObject, QThread, pyqtSignal
import time
import os

from scripts.NNModel import NNModel


class myGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Python Project Build v1.0 [Personal Repo]')
        self.setGeometry(300, 300, 600, 600) # mixture of move(x, y) and resize(width, height)

        # create instance of our NNModel Class to download dataset 
        global nnmodel
        nnmodel = NNModel()

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

        # Add view menu
        viewmenu = menubar.addMenu('&View')
        # Version History
        version_history = QMenu('Version history', self)
        version_number = QAction('Version 1.1', self)
        version_history.addAction(version_number)
        viewmenu.addMenu(version_history)


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
        self.addToolBar(Qt.TopToolBarArea, fileToolBar)
        #fileToolBar = self.addToolBar('Tab control')    # set name of toolbar
        fileToolBar.addAction(remove_all_tab_menu)
        fileToolBar.addAction(remove_tab_menu)
        
        # Additional toolbar to left side of window
        sideToolBar = QToolBar(self)
        self.addToolBar(Qt.LeftToolBarArea, sideToolBar)
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

    # ====== open/close tabs ======
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
        self.table_widget.tabs.insertTab(current_index, tab_2_widget(), "Tab 2")
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

# Tab 1 widget for welcome information
class tab_1_widget(QWidget):
    def __init__(self, parent=None):
        super(tab_1_widget, self).__init__(parent)
        self.main_layout = QVBoxLayout(self)
        text_label = QLabel("This is Tab 1")
        self.main_layout.addWidget(text_label)

# Tab 2 widget for drawing canvas
class tab_2_widget(QWidget):
    def __init__(self):
        super(tab_2_widget, self).__init__()

        # upper layout
        upper_layout = QHBoxLayout(self)

        # set up layout for canvas 
        self.canvas_layout = QVBoxLayout(self)
        self.canvas_label = QLabel()
        text_label = QLabel("This is Tab 2: Drawing Canvas")
        self.canvas_layout.addWidget(text_label)
        self.canvas = QtGui.QPixmap(400, 400)
        self.canvas.fill(QtGui.QColor('white'))
        self.canvas_label.setPixmap(self.canvas)
        self.canvas_layout.addWidget(self.canvas_label)

        # Test layout
        self.test_layout = QVBoxLayout(self)
        self.button1 = QPushButton("Load Model", self)
        #self.button1.clicked.connect(self.load_selected_model)
        self.button2 = QPushButton("Predict", self)

        #self.button2.clicked.connect(self.predict_show)

        # Set up text for text browser
        prediction_text = "Prediction: "
        Accuracy_text = "Confidence: "
        self.selected_model_text = "N/A"

        # Text browsor for currently loaded model
        self.set_model_browser = QTextBrowser(self)
        self.set_model_browser.setText("Model loaded: " + self.selected_model_text)
        self.set_model_browser.setFont(QFont('Serif', 10))
        self.set_model_browser.setMaximumHeight(50)

        # Text browser for prediction
        self.prediction_browser = QTextBrowser(self)
        self.prediction_browser.append(prediction_text)
        self.prediction_browser.setFont(QFont('Serif', 10))
        self.prediction_browser.setMaximumHeight(50)

        # Test browser for accuracy
        self.accuracy_browser = QTextBrowser(self)
        self.accuracy_browser.setText(Accuracy_text)
        self.accuracy_browser.setFont(QFont('Serif', 10))
        self.accuracy_browser.setMaximumHeight(50)

        # Add combobox to choose model
        self.model_combo = QComboBox(self)
        model_option_array = ["Select", "Default_Net", "CNN_Net", "ResNet_Net", "Custom_Net"]
        self.model_combo.addItems(model_option_array)

        # Layout setup
        self.test_layout.addSpacing(50)
        self.test_layout.addWidget(self.set_model_browser)
        self.test_layout.addSpacing(10)
        self.test_layout.addWidget(self.prediction_browser)
        self.test_layout.addSpacing(10)
        self.test_layout.addWidget(self.accuracy_browser)
        self.test_layout.addWidget(self.model_combo)
        self.test_layout.addWidget(self.button1)
        self.test_layout.addWidget(self.button2)
        self.test_layout.addSpacing(150)

        # add nested layout
        upper_layout.addLayout(self.canvas_layout)
        #upper_layout.addStretch()   # prevents button from stretching when window size is changed
        upper_layout.addLayout(self.test_layout)
        
        
        self.last_x = None
        self.last_y = None

        # add clear button to remove drawn image from canvas
        clearButton = QPushButton('Clear')
        self.canvas_layout.addWidget(clearButton)
        clearButton.clicked.connect(self.clear_click)

        # add paint brush thickeness selection widget
        self.painter_size_combo = QComboBox(self)
        self.canvas_layout.addWidget(self.painter_size_combo)
        painter_option_array = ["Size 3", "Size 6", "Size 9", "Size 12", "Size 18"]
        self.painter_size_combo.addItems(painter_option_array)





        #self.text_label_height = 20 # used to offset cursor
        #print(self.text_label_height)

        #self.text_box_height = self.model_combo.height() # used to offset cursor
        
        # total offset from widgets, added to y-position
        #self.total_offset = self.text_label_height + self.text_box_height

        # add second layout to main outer layout
        #self.layout_outer.addLayout(self.canvas_layout2)

    # check size of paint
    def check_paint_size(self, event: QtGui.QMouseEvent):
        # retrieve current text displayed in combo box
        text = self.painter_size_combo.currentText()

        # check which option is selected 
        if text == "Size 3":
            paint_width = 3
            self.last_x = event.x()
            self.last_y = event.y()
        elif text == "Size 6":
            paint_width = 6
            self.last_x = event.x()
            self.last_y = event.y()
        elif text == "Size 9":
            paint_width = 9
            self.last_x = event.x()
            self.last_y = event.y()
        elif text == "Size 12":
            paint_width = 12
            self.last_x = event.x()
            self.last_y = event.y()
        elif text == "Size 18":
            paint_width = 18
            self.last_x = event.x()
            self.last_y = event.y()

        return paint_width

    # clear canvas
    def clear_click(self):
        self.canvas_label.setPixmap(self.canvas)




    # resize event of canvas
    #def resizeEvent(self, e):
    #    canvas_resize = QtGui.QPixmap(500, 500)
    #    canvas_resize.fill(QtGui.QColor('white'))
    #    self.canvas = canvas_resize.scaled(self.width(), self.height())
    #    self.canvas_label.setPixmap(self.canvas)
    #    self.canvas_label.resize(self.width(), self.height())


    # Adapted from https://www.pythonguis.com/tutorials/bitmap-graphics/
    # set up painter and drawline lines based on cursor coordinates at each detected mouse movement
    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self.last_x is None: 
            self.last_x = event.x()
            self.last_y = event.y()
            return 

        self.painter = QtGui.QPainter(self.canvas_label.pixmap())
        self.p = self.painter.pen()
        self.p.setWidth(self.check_paint_size(event))
        self.painter.setPen(self.p)

        self.painter.drawLine(self.last_x, self.last_y, event.x(), event.y())
        self.painter.end()
        self.update()

        self.last_x = event.x()
        self.last_y = event.y()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        # Reset when mouse click released
        self.last_x = None
        self.last_y = None

        # Save drawn image
        imgage = QtGui.QPixmap(self.canvas_label.pixmap())
        imgage.save("images\loadedimage.png")



# Tab 3 will display the View Training Images
class tab_3_widget(QWidget):
    def __init__(self, parent = None):
        super(tab_3_widget, self).__init__(parent)
        self.initUI()

    def initUI(self):

        # OUTER BOX HOLDING EVERYTHING TOGETHER
        self.layout_outer = QHBoxLayout()
        self.setLayout(self.layout_outer)        
        
        # self.images is a grid of 100 x 100 images from the dataset
        self.page = 0   # This is the page number
       
        # HOLD IMAGE VIEWER + NAV BUTTONS
        self.scroll_nav_layout = QVBoxLayout()

        # add scroll component
        self.images = QWidget()
        self.grid = QGridLayout()

        # scroll settings
        self.images.setLayout(self.grid)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.images)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumWidth(450)
        self.scroll_nav_layout.addWidget(self.scroll)
  
        # NAVIGATE LAYOUT TO HOLD NEXT AND BACK BUTTONS
        self.navigate_layout = QHBoxLayout()
        back_button = QPushButton("Previous")
        next_button =  QPushButton("Next")

        self.navigate_layout.addWidget(back_button)
        self.navigate_layout.addWidget(next_button)
        back_button.clicked.connect(self.load_prev_page)
        next_button.clicked.connect(self.load_next_page)
        
        # RIGHT SIDE OF OVERALL LAYOUT
        # Page counter
        self.right_layout = QVBoxLayout()
        text_box = QTextBrowser(self)
        text_message = "Welcome to MNIST Dataset Training image viewer"
        text_box.setText(text_message)
        self.right_layout.addWidget(text_box)

        stats_button = QPushButton("See Statistics", self)
        self.right_layout.addWidget(stats_button)

        self.right_layout.addSpacing(200)

        self.page_num_box = QTextBrowser(self)
        self.page_num_box.setMaximumHeight(30)
        self.page_number = str(self.page)
        self.page_num_box.setText("Page: "+self.page_number)
        self.right_layout.addWidget(self.page_num_box)

        ####### CONFIGURE LAYOUTS ######
        # add grid and nav to scroll_nav_layout
        self.scroll_nav_layout.addLayout(self.grid)
        self.scroll_nav_layout.addLayout(self.navigate_layout)

        # add above to outer grid
        self.layout_outer.addLayout(self.scroll_nav_layout)
        self.layout_outer.addLayout(self.right_layout)

        self.load_dataset_images()

    ###### Functions for grid image control ######
    def load_next_page(self):
            self.page = self.page + 1
            self.load_dataset_images()
            self.page_num_box.setText("Page: "+ str(self.page))

    def load_prev_page(self):
        # ensure page number is not negative
        if self.page > 0:
            self.page = self.page - 1
            self.load_dataset_images()
            self.page_num_box.setText("Page: "+ str(self.page))

    # use nested for loop to iterate each row and column of our image grid and add a pixmap image
    def load_dataset_images(self):
            for row in range(0,50):    # range(0,50) number of rows per page --> 50 rows
                for col in range(0,4):    # range(0,4) number of columns per page --> 4 columns

                    # each image grid will display 4 x 50 = 200 images
                    label = QLabel()

                    #value = ['1']
                    #value_iter = iter(nnmodel.train_dataset)
                    #current_value = next(value_iter)
                    #print(current_value) 

                    # max images per grid = 50 * 4 = 200
                    image_array = np.squeeze(nnmodel.train_dataset[10*row+col+200*self.page][0])

                    # save as grayscale images
                    plot.imsave("images\\temp_grid_image.png", image_array, cmap='gray')
                    saved_image = QtGui.QPixmap("images\\temp_grid_image.png")
                    
                    # upscale the images
                    saved_image_scaled = saved_image.scaled(100, 100, Qt.KeepAspectRatio, Qt.FastTransformation)

                    label.setPixmap(saved_image_scaled)
                    self.grid.addWidget(label, row, col)

# Tab 4 will display the View Testing Images
class tab_4_widget(QWidget):
    def __init__(self, parent = None):
        super(tab_4_widget, self).__init__(parent)
        self.initUI()

    def initUI(self):

        # OUTER BOX HOLDING EVERYTHING TOGETHER
        self.layout_outer = QHBoxLayout()
        self.setLayout(self.layout_outer)        
        
        # self.images is a grid of 100 x 100 images from the dataset
        self.page = 0   # This is the page number
       
        # HOLD IMAGE VIEWER + NAV BUTTONS
        self.scroll_nav_layout = QVBoxLayout()

        # add scroll component
        self.images = QWidget()
        self.grid = QGridLayout()

        # scroll settings
        self.images.setLayout(self.grid)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.images)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumWidth(450)
        self.scroll_nav_layout.addWidget(self.scroll)
  
        # NAVIGATE LAYOUT TO HOLD NEXT AND BACK BUTTONS
        self.navigate_layout = QHBoxLayout()
        back_button = QPushButton("Previous")
        next_button =  QPushButton("Next")

        self.navigate_layout.addWidget(back_button)
        self.navigate_layout.addWidget(next_button)
        back_button.clicked.connect(self.load_prev_page)
        next_button.clicked.connect(self.load_next_page)
        
        # RIGHT SIDE OF OVERALL LAYOUT
        # Page counter
        self.right_layout = QVBoxLayout()
        text_box = QTextBrowser(self)
        text_message = "Welcome to MNIST Dataset Testing image viewer"
        text_box.setText(text_message)
        self.right_layout.addWidget(text_box)

        stats_button = QPushButton("See Statistics", self)
        self.right_layout.addWidget(stats_button)

        self.right_layout.addSpacing(200)

        self.page_num_box = QTextBrowser(self)
        self.page_num_box.setMaximumHeight(30)
        self.page_number = str(self.page)
        self.page_num_box.setText("Page: "+self.page_number)
        self.right_layout.addWidget(self.page_num_box)

        ####### CONFIGURE LAYOUTS ######
        # add grid and nav to scroll_nav_layout
        self.scroll_nav_layout.addLayout(self.grid)
        self.scroll_nav_layout.addLayout(self.navigate_layout)

        # add above to outer grid
        self.layout_outer.addLayout(self.scroll_nav_layout)
        self.layout_outer.addLayout(self.right_layout)

        self.load_dataset_images()

    ###### Functions for grid image control ######
    def load_next_page(self):
            self.page = self.page + 1
            self.load_dataset_images()
            self.page_num_box.setText("Page: "+ str(self.page))

    def load_prev_page(self):
        # ensure page number is not negative
        if self.page > 0:
            self.page = self.page - 1
            self.load_dataset_images()
            self.page_num_box.setText("Page: "+ str(self.page))

    # use nested for loop to iterate each row and column of our image grid and add a pixmap image
    def load_dataset_images(self):
            for row in range(0,50):    # range(0,50) number of rows per page --> 50 rows
                for col in range(0,4):    # range(0,4) number of columns per page --> 4 columns

                    # each image grid will display 4 x 50 = 200 images
                    label = QLabel()

                    # max images per grid = 50 * 5 = 250
                    image_array = np.squeeze(nnmodel.test_dataset[row+col+200*self.page][0])

                    # save as grayscale images
                    plot.imsave("images\\temp_grid_image.png", image_array, cmap='gray')
                    saved_image = QtGui.QPixmap("images\\temp_grid_image.png")
                    
                    # upscale the images
                    saved_image_scaled = saved_image.scaled(100, 100, Qt.KeepAspectRatio, Qt.FastTransformation)

                    label.setPixmap(saved_image_scaled)
                    self.grid.addWidget(label, row, col)
                
    
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

        download_button = QPushButton('Download MNIST Dataset', self)
        download_button.clicked.connect(self.print_to_textBrowser)  # link to signal
        self.layout.addWidget(download_button)

        train_button = QPushButton('Train', self)
        self.layout.addWidget(train_button)

        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.cancel_textBrowser)
        self.layout.addWidget(cancel_button)


    
    # test functions
    def testFunction(self):
        print('Sucess')

    # in the event the download MNIST dataset has been pressed
    def print_to_textBrowser(self):
        download_message = "Download button has been pressed"
        self.text_box.setText(download_message)  # add line of text below previous text
        nnmodel.downloadTrainingData()
        self.text_box.append("Downloading Training Dataset...")
        nnmodel.downloadTestData()
        self.text_box.append("Downloading Testing Dataset...")
        self.text_box.append("Download Finished!")
        progress_bar_window().exec()
        

    def cancel_textBrowser(self):
        cancel_message = "Process Cancelled"
        self.text_box.setText(cancel_message)   # replace all existing text with new text

# progress bar window for when downloading dataset
class progress_bar_window(QDialog):
    def __init__(self):
        super(QDialog, self).__init__()
        self.initUI()

    def initUI(self):
        self.layout2 = QVBoxLayout()
        self.setLayout(self.layout2)
        self.setGeometry(400, 400, 300, 100)
        self.setWindowTitle("Downloads")
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimum(0)
        self.layout2.addWidget(self.progress_bar)



        



