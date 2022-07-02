# File name: myGUI.py
# Author: Francis Cho
# Project Version: 1.0
# Description: GUI for the Handwritten Digit/Letter Recognizer using PyQt5
# Python Version: 3.1

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

import cv2

from scripts.NNModel import NNModel


class myGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Python Project Build v1.0 [Personal Repo]')
        self.setGeometry(300, 300, 650, 650) # mixture of move(x, y) and resize(width, height)

        # create instance of our NNModel Class to download dataset 
        global nnmodel
        nnmodel = NNModel()

        # Initialize tab widget
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
        self.add_tab_1()    # add welcome tab on launch

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
        # Tab 1
        new_tab_menu = QAction('Tab 1: Welcome', self)
        filemenu.addAction(new_tab_menu)
        new_tab_menu.triggered.connect(self.add_tab_1)  # link to signal

        # Tab 2
        new_tab_menu2 = QAction('Tab 2: Draw Canvas', self)
        filemenu.addAction(new_tab_menu2)
        new_tab_menu2.triggered.connect(self.add_tab_2) # link to signal

        # Tab 3
        new_tab_menu3 = QAction('Tab 3: Import Dataset', self)
        filemenu.addAction(new_tab_menu3)
        new_tab_menu3.triggered.connect(self.add_tab_3) # link to signal

        # Tab 4
        new_tab_menu4 = QAction('Tab 4: View Image Train', self)
        filemenu.addAction(new_tab_menu4)
        new_tab_menu4.triggered.connect(self.add_tab_4) # link to signal

        # Tab 5
        new_tab_menu5 = QAction('Tab 5: View Image Test', self)
        filemenu.addAction(new_tab_menu5)
        new_tab_menu5.triggered.connect(self.add_tab_5) # link to signal

        # Tab 6
        new_tab_menu6 = QAction('Tab 6: Video Camera', self)
        filemenu.addAction(new_tab_menu6)
        new_tab_menu6.triggered.connect(self.add_tab_6) # link to signal


        # close current tab
        remove_tab_menu = QAction('Close Current Tab', self)
        filemenu.addAction(remove_tab_menu)
        remove_tab_menu.triggered.connect(self.removeTab)

        # close all open tabs
        remove_all_tab_menu = QAction('Close All Tabs', self)
        filemenu.addAction(remove_all_tab_menu)
        remove_all_tab_menu.triggered.connect(self.removeAllTabs)

        # set a status bar at bottom of window
        self.statusBar().showMessage('Ready')
        
        # ====== Add toolbars ======
        fileToolBar = self.addToolBar('Tab control top')
        self.addToolBar(Qt.TopToolBarArea, fileToolBar)
        fileToolBar.addAction(remove_all_tab_menu)
        fileToolBar.addAction(remove_tab_menu)
        
        # Additional toolbar to left side of window
        sideToolBar = self.addToolBar('Tab control side')
        self.addToolBar(Qt.LeftToolBarArea, sideToolBar)
        sideToolBar.addAction(new_tab_menu)
        sideToolBar.addAction(new_tab_menu2)
        sideToolBar.addAction(new_tab_menu3)
        sideToolBar.addAction(new_tab_menu4)
        sideToolBar.addAction(new_tab_menu5)
        sideToolBar.addAction(new_tab_menu6)

        self.show()  # make visible

    # ====== open/close tabs ======
    # when tab is opened automatically switches to new opened tab
    def add_tab_1(self):
        # saves current index our main window is open at
        current_index = self.table_widget.tabs.currentIndex()
        # opens new tab at that particular index
        self.table_widget.tabs.insertTab(current_index, tab_1_widget(), "Tab 1: Welcome")
        # sets current index to the index of newly opened tab
        self.table_widget.tabs.setCurrentIndex(current_index)

    def add_tab_2(self):
        current_index = self.table_widget.tabs.currentIndex()
        self.table_widget.tabs.insertTab(current_index, tab_2_widget(), "Tab 2: Drawing Canvas")
        self.table_widget.tabs.setCurrentIndex(current_index)

    def add_tab_3(self):
        current_index = self.table_widget.tabs.currentIndex()
        self.table_widget.tabs.insertTab(current_index, tab_3_widget(), "Tab 3: Import and Train Dataset")
        self.table_widget.tabs.setCurrentIndex(current_index)

    def add_tab_4(self):
        current_index = self.table_widget.tabs.currentIndex()
        self.table_widget.tabs.insertTab(current_index, tab_4_widget(), "Tab 4: View Training Images")
        self.table_widget.tabs.setCurrentIndex(current_index)

    def add_tab_5(self):
        current_index = self.table_widget.tabs.currentIndex()
        self.table_widget.tabs.insertTab(current_index, QWidget(), "Tab 5: View Testing Images")
        self.table_widget.tabs.setCurrentIndex(current_index)

    def add_tab_6(self):
        current_index = self.table_widget.tabs.currentIndex()
        self.table_widget.tabs.insertTab(current_index, tab_6_widget(), "Tab 6: Camera Video")
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
    def __init__(self):
        super(tab_1_widget, self).__init__()
        self.tab_1_layout = QVBoxLayout(self)
        title_label = QLabel("Handwritten Digit/Letter Recogniser Python Project")
        title_label.setFont(QFont('Serif', 16))
        self.tab_1_layout.addWidget(title_label)
        author_label = QLabel("Created by Francis Cho")
        self.tab_1_layout.addWidget(author_label)

        self.welcome_text_box = QTextBrowser(self)
        self.welcome_text_box.setText(" ")
        self.welcome_text_box.append("Project Version 1.0")
        self.welcome_text_box.append(" ")
        self.welcome_text_box.append("Features: ")
        self.welcome_text_box.append("- Drawing canvas")
        self.welcome_text_box.append(" ")
        self.welcome_text_box.append("- EMNIST Train/Test Dataset Viewer")
        self.welcome_text_box.append(" ")
        self.welcome_text_box.append("- 3 Pre-trained NN Models:")
        self.welcome_text_box.append("      - Default_Net")
        self.welcome_text_box.append("      - CNN_Net")
        self.welcome_text_box.append("      - ResNet_Net")
        self.welcome_text_box.append(" ")
        self.welcome_text_box.append("- Custom model training with model and epoch selection")
        self.welcome_text_box.append(" ")
        self.welcome_text_box.append("- Detect handwriting through camera image capture")
        self.welcome_text_box.setFont(QFont('Serif', 10))
        
        self.tab_1_layout.addWidget(self.welcome_text_box)
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
        self.button1.clicked.connect(self.load_selected_model)
        self.button2 = QPushButton("Predict", self)

        self.button2.clicked.connect(self.predict_show)

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

    def predict_show(self):
        self.prediction, self.accuracy = nnmodel.process_input_image()
        self.prediction_browser.setText("Prediction is: " + self.prediction)
        self.prediction_browser.setFont(QFont('Serif', 10))
        self.accuracy_browser.setText("Confidence: " + self.accuracy + "%")
        self.accuracy_browser.setFont(QFont('Serif', 10))
     
    # loads from saved model
    def load_selected_model(self):
        # retrieve current text displayed in combo box to get model names
        chosen_model = self.model_combo.currentText()

        # check which model is chosen 
        if (chosen_model == 'Default_Net'):
            # load the chosen model
            nnmodel.load_model_1()
            self.selected_model_text = 'Default_Net'
            self.set_model_browser.setText("Model loaded: " + self.selected_model_text)
        elif (chosen_model == 'CNN_Net'):
            nnmodel.load_model_2()
            self.selected_model_text = 'CNN_Net'
            self.set_model_browser.setText("Model loaded: " + self.selected_model_text)
        elif (chosen_model == 'ResNet_Net'):
            nnmodel.load_model_3()
            self.selected_model_text = 'ResNet_Net'
            self.set_model_browser.setText("Model loaded: " + self.selected_model_text)
        elif (chosen_model == 'Custom_Net'):
            
            self.selected_model_text = 'Custom_Net'

            # check which model to load based on what the custom model was trained on
            model_type =  str(type(nnmodel.model).__name__)
            if (model_type == 'Default_Net' and int(nnmodel.load_model_4_Default()) != 0):
                nnmodel.load_model_4_Default()
                self.set_model_browser.setText("Model loaded: " + self.selected_model_text) 
    
            elif (model_type == 'CNN_Net' and int(nnmodel.load_model_4_Default()) != 0):
                nnmodel.load_model_4_CNN
                self.set_model_browser.setText("Model loaded: " + self.selected_model_text) 

            elif(model_type == 'ResNet_Net' and int(nnmodel.load_model_4_Default()) != 0):
                nnmodel.load_model_4_ResNet
                self.set_model_browser.setText("Model loaded: " + self.selected_model_text) 

            else:
                self.set_model_browser.setText("Model does not exist: " + self.selected_model_text) 


            #self.set_model_browser.setText("Model loaded: " + self.selected_model_text)       
        #self.prediction_browser.append(chosen_model + ' model loaded!')

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


# Tab 3 widget for importing and training dataset
class tab_3_widget(QWidget):
        def __init__(self):
            super(tab_3_widget, self).__init__()
            self.tab_3_layout = QVBoxLayout(self)
            text_label = QLabel("This is Tab 3: Import Datasets")
            self.tab_3_layout.addWidget(text_label)

            self.text_box = QTextBrowser(self)
            welcome_message = "Welcome! Please download/load the EMNIST dataset to begin."
            self.tab_3_layout.addWidget(self.text_box)
            self.text_box.setText(welcome_message)

            download_button = QPushButton('Download/Load EMNIST Dataset', self)
            download_button.clicked.connect(self.download_with_thread)  # link to signal with thread
            self.tab_3_layout.addWidget(download_button)

            instruction_text = QLabel("Select a model to train: (NOTE: Resulting trained model will be saved as 'Custom_Net')")
            self.tab_3_layout.addWidget(instruction_text)

            self.model_combo_box = QComboBox(self)
            self.tab_3_layout.addWidget(self.model_combo_box)
            model_option_array = ["Select", "Default_Net", "CNN_Net", "ResNet_Net"]
            self.model_combo_box.addItems(model_option_array)

            load_model_button = QPushButton('Select Model', self)
            load_model_button.clicked.connect(self.set_selected_model)
            self.tab_3_layout.addWidget(load_model_button)

            instruction_text_epoch = QLabel("Select number of epoch: ")
            self.tab_3_layout.addWidget(instruction_text_epoch)

            # add combo box for number of epochs to be selected 
            self.epoch_combo_box = QComboBox(self)
            self.tab_3_layout.addWidget(self.epoch_combo_box)
            epoch_option_array = ["1", "2", "3", "4", "5", "6", "7", "8"]
            self.epoch_combo_box.addItems(epoch_option_array)

            # add button to select the chossen number of epoch
            load_epoch_button = QPushButton('Select Epoch', self)
            load_epoch_button.clicked.connect(self.set_selected_epoch)
            self.tab_3_layout.addWidget(load_epoch_button)

            instruction_text_train = QLabel("Train & Delete Model: ")
            self.tab_3_layout.addWidget(instruction_text_train)

            # Train the model
            train_button = QPushButton('Train', self)
            self.tab_3_layout.addWidget(train_button)
            train_button.clicked.connect(self.train_with_thread)    # link to signal with thread

            # Delete Custom_Model
            delete_model_button = QPushButton("Delete Model (Custom_Net)", self)
            self.tab_3_layout.addWidget(delete_model_button)
            delete_model_button.clicked.connect(self.remove_model_file)

            cancel_button = QPushButton('Cancel')
            cancel_button.clicked.connect(self.cancel_textBrowser)
            self.tab_3_layout.addWidget(cancel_button)

        # function to remove the Custom_Net file
        def remove_model_file(self):
            file_name = 'saved_models\Custom_Net'
            if os.path.exists(file_name) and not os.path.isdir(file_name) and not os.path.islink(file_name):
                os.remove(file_name)
            else:
                print("File does not exist")
                self.text_box.append("File does not exist")

        # function to set the number of epochs
        def set_selected_epoch(self):
            self.chosen_epoch = int(self.epoch_combo_box.currentText())    # save as int
            nnmodel.save_epoch_number(self.chosen_epoch)
            

        # Set the model we want to train the dataset on
        def set_selected_model(self):
            # retrieve current text displayed in combo box to get model names
            self.chosen_model = self.model_combo_box.currentText()

            # check which model is chosen
            if (self.chosen_model == 'Select'):
                self.text_box.setText("That is not a valid model! Please try again.")

            elif (self.chosen_model == 'Default_Net'):
                # load the chosen model
                nnmodel.set_model_to_train('Default_Net')
                self.text_box.setText(self.chosen_model + ' model set to train!')

            elif (self.chosen_model == 'CNN_Net'):
                nnmodel.set_model_to_train('CNN_Net')
                self.text_box.setText(self.chosen_model + ' model set to train!')

            elif (self.chosen_model == 'ResNet_Net'):
                nnmodel.set_model_to_train('ResNet_Net')
                self.text_box.setText(self.chosen_model + ' model set to train!')


        def cancel_textBrowser(self):
            cancel_message = "Process Cancelled. Please wait a moment."
            self.text_box.setText(cancel_message)   # replace all existing text with new text
            nnmodel.check_cancel(True)

        def download_with_thread(self):
            self.text_box.setText("Download button has been pressed")  # add line of text below previous text

            self.thread = QThread(parent = self)
            self.worker = worker()

            self.worker.moveToThread(self.thread)

            # Connect signals and slots
            self.thread.started.connect(self.worker.worker_download_dataset)
            self.worker.progress.connect(self.reportProgress)
            self.worker.progress.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            # Start the thread
            self.thread.start()

            self.text_box.append("Download Finished!")
            

        # Adapted from https://realpython.com/python-pyqt-qthread/
        def train_with_thread(self):
    
            self.text_box.append("Begin Training Dataset...")
            # Need to set parent = self or else destoryed thread will continue to run after cancelled
            self.thread = QThread(parent = self)    # create QThread object
            self.worker = worker()                  # create worker object

            # Move worker to the thread
            self.worker.moveToThread(self.thread)

            # Connect signals and slots
            self.thread.started.connect(self.worker.worker_run_train)
            self.worker.progress.connect(self.reportProgress)
            self.worker.epoch_progress.connect(self.report_epoch_progress)
            self.worker.progress.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            # Start the thread
            self.thread.start()

        # used to display error message to text browser
        def reportProgress(self, int):
            if int == 1:
                self.text_box.setText("EMNIST Dataset is missing! Please download and try again.")

            elif int == 2:
                self.text_box.append("Starting EMNIST Training dataset download...")

            elif int == 3:
                self.text_box.append("Finished downloading EMNIST Training dataset.")
            
            elif int == 4:
                self.text_box.append("Starting EMNIST Testing dataset download...")

            elif int == 5:
                self.text_box.append("Finished downloading EMNIST Testing dataset.")


        def report_epoch_progress(self, input):
            if input == 'epoch_1':
                self.text_box.append("===========================")
                self.text_box.append("Training Epoch: 1")

            elif input == 'epoch_2':
                self.text_box.append("Training Epoch: 2")

            elif input == 'epoch_3':
                self.text_box.append("Training Epoch: 3")

            elif input == 'epoch_4':
                self.text_box.append("Training Epoch: 4")

            elif input == 'epoch_5':
                self.text_box.append("Training Epoch: 5")

            elif input == 'epoch_6':
                self.text_box.append("Training Epoch: 6")

            elif input == 'epoch_7':
                self.text_box.append("Training Epoch: 7")

            elif input == 'epoch_8':
                self.text_box.append("Training Epoch: 8")                

            elif input == 'epoch_complete':
                self.text_box.append("===========================")
                self.text_box.append("Model training completed.")

            elif input == 'epoch_cancelled':
                self.text_box.append("===========================")
                self.text_box.append("Model training cancelled.")
    


# Need to use QThread to prevent GUI from freezing during dataset training
# Adapted from https://realpython.com/python-pyqt-qthread/
# create worker class 
class worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    epoch_progress = pyqtSignal(str)

    def worker_run_train(self):
        nnmodel.check_cancel(False)

        #nnmodel.save_epoch_number(1)    # set default option to epoch 1
        #epoch_range = nnmodel.epoch_number + 1

        try:
            epoch_range = nnmodel.epoch_number + 1
            # Adapted from COMPSYS 302 PyTorch lab
            # print to terminal to check progress
            since = time.time()
            for epoch in range(1, epoch_range):
                # check if user has cancelled the process
                if (nnmodel.cancel == True):
                    self.finished.emit()
                    self.epoch_progress.emit('epoch_cancelled')
                    break

                # emit signal for each epoch being trained up to epoch 8
                self.epoch_progress.emit(nnmodel.check_epoch(epoch))

                epoch_start = time.time()
                nnmodel.train_model(epoch)

                # Testing purposes, TODO REMOVE IN REFACTOR
                print(type(nnmodel.model).__name__)

                m, s = divmod(time.time() - epoch_start, 60)
                print(f'Training time: {m:.0f}m {s:.0f}s')
                nnmodel.test_model()
                m, s = divmod(time.time() - epoch_start, 60)
                print(f'Testing time: {m:.0f}m {s:.0f}s')


            m, s = divmod(time.time() - since, 60)
            print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {nnmodel.device}!')

            self.epoch_progress.emit('epoch_complete')
            self.progress.emit(20)
            self.finished.emit()

            # check which model to save
            # save the model "torch.save(model.state_dict(), PATH)"
            #torch.save(nnmodel.model.state_dict(), 'saved_models\CNN_Net')
            #torch.save(nnmodel.model.state_dict(), 'saved_models\Default_Net')
            #torch.save(nnmodel.model.state_dict(), 'saved_models\ResNet_Net')
            torch.save(nnmodel.model.state_dict(), 'saved_models\Custom_Net')

        except AttributeError:
            print("EMNIST Dataset Missing!")
            self.progress.emit(1)


    def worker_download_dataset(self):
        nnmodel.check_cancel(False)
        if (nnmodel.cancel == True):
            self.finished.emit()
        else:
            self.progress.emit(2)
            nnmodel.download_training_dataset()
            nnmodel.load_training_dataset()
            self.progress.emit(3)   # emit 2 if finished downloading/loading train data

            self.progress.emit(4)
            nnmodel.download_testing_dataset()
            nnmodel.load_testing_dataset()
            self.progress.emit(5)   # emit 3 if finished downloading/loading test data

            self.finished.emit()
                
    
# Tab 4 widget for viewing training images
class tab_4_widget(QWidget):
    def __init__(self):
        super(tab_4_widget, self).__init__()
        self.initUI()

    def initUI(self):

        # Outer layout holds every other layout together
        self.layout_outer = QHBoxLayout()
        self.setLayout(self.layout_outer)  

        # initialize page number
        self.page_number = 0      
        
        # This layout holds the image display grid
        self.images_layout = QVBoxLayout()

        # add scroll component
        self.images_widget = QWidget()
        self.image_grid = QGridLayout()

        # configure scroll settings
        self.images_widget.setLayout(self.image_grid)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.images_widget)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumWidth(450 )
        self.images_layout.addWidget(self.scroll)

        # add layout to hold next/back buttons
        self.navigate_layout = QHBoxLayout()
        back_button = QPushButton("Previous", self)
        next_button = QPushButton("Next", self)

        # add widgets to layout
        self.navigate_layout.addWidget(back_button)
        self.navigate_layout.addWidget(next_button)

        # connect to signal
        back_button.clicked.connect(self.load_back_page)
        next_button.clicked.connect(self.load_next_page)

        # Right side of overall layout setup
        self.right_layout = QVBoxLayout()
        self.text_box = QTextBrowser(self)
        text_message = "EMNIST Dataset Train Image Viewer"
        self.text_box.setText(text_message)
        self.text_box.append(" ")
        self.text_box.append("200 Images loaded per page")
        self.right_layout.addWidget(self.text_box)

        # simple stats button
        stats_button = QPushButton("See Statistics", self)
        self.right_layout.addWidget(stats_button)
        # TODO connect stats_button signal

        # display current page number
        self.page_num_box = QTextBrowser(self)
        self.page_num_box.setMaximumHeight(30)
        self.page_num_box.setText("Page: " + str(self.page_number))
        self.right_layout.addWidget(self.page_num_box)
  
        #====== CONFIGURE LAYOUTS ======
        # add image grid to the image layout
        self.images_layout.addLayout(self.image_grid)
        self.images_layout.addLayout(self.navigate_layout)

        # add above to outer layout containing everything
        self.layout_outer.addLayout(self.images_layout)
        self.layout_outer.addLayout(self.right_layout)

        self.load_dataset_images()


    #====== CONFIGURE FUNCTIONS ======
    def load_next_page(self):
            self.page_number = self.page_number + 1
            self.load_dataset_images()
            self.page_num_box.setText("Page: "+ str(self.page_number))  
            self.scroll.verticalScrollBar().setValue(0)     # reset scroll to top

    def load_back_page(self):
        # ensure page number is not negative
        if self.page_number > 0:
            self.page_number = self.page_number - 1
            self.load_dataset_images()
            self.page_num_box.setText("Page: "+ str(self.page_number))
            self.scroll.verticalScrollBar().setValue(0)     # reset scroll to top

    # use nested for loop to iterate each row and column of our image grid and add a pixmap image
    def load_dataset_images(self):

        try:
            # set up row and columns of our image grid
            self.max_row = 50
            self.max_col = 4
            self.max_images = self.max_row * self.max_col

            for row in range(0, self.max_row):    # range(0,50) number of rows per page --> 50 rows
                for col in range(0, self.max_col):    # range(0,4) number of columns per page --> 4 columns

                    # each image grid will display 4 x 50 = 200 images
                    label = QLabel()

                    self.dataset_index = 10*row + col + self.max_images*self.page_number

                    # use squeeze to remove all dimensions of length 1
                    processed_image = np.squeeze(nnmodel.train_dataset[self.dataset_index][0])
                    processed_image = np.fliplr(processed_image)
                    processed_image = np.rot90(processed_image)

                    file_name = './images/temp_grid_image.png'
                    plot.imsave(file_name, processed_image, cmap = 'gray')

                    saved_image = QtGui.QPixmap(file_name)

                    # upscale the image for better presentation
                    saved_image_scaled = saved_image.scaled(100, 100, Qt.KeepAspectRatio, Qt.FastTransformation)

                    label.setPixmap(saved_image_scaled)
                    self.image_grid.addWidget(label, row, col) 

        except AttributeError:
            print("EMNIST Dataset missing!")
            self.text_box.setText('EMNIST Dataset missing!')
            self.text_box.append(' ')
            self.text_box.append('Please download EMNIST Dataset and try again.')


class tab_6_widget(QWidget):
    def __init__(self):
        super(tab_6_widget, self).__init__()
        self.initUI()

    def initUI(self):
        # upper layout
        upper_layout = QHBoxLayout(self)

        # set up layout for canvas 
        self.canvas_layout = QVBoxLayout(self)
        self.canvas_label = QLabel()
        text_label = QLabel("This is Tab 6: Predict with captured image")
        self.canvas_layout.addWidget(text_label)
        self.canvas = QtGui.QPixmap(400, 400)
        self.canvas.fill(QtGui.QColor('white'))
        self.canvas_layout.addWidget(self.canvas_label)

        # Test layout
        self.test_layout = QVBoxLayout(self)
        self.button1 = QPushButton("Load Model", self)
        self.button1.clicked.connect(self.load_selected_model)
        self.button2 = QPushButton("Predict", self)
        self.button2.clicked.connect(self.predict_show)

        # Add button to start live video feed
        self.openCameraButton = QPushButton('Open Camera')
        self.clearButton = QPushButton('Clear')
        self.openCameraButton.clicked.connect(self.camera_with_thread)

        # Add button to take photo 
        #self.takePhotoButton = QPushButton('Take Photo')
        #self.takePhotoButton.clicked.connect(self.take_photo)

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
        self.camera_setting_text = QLabel('Camera Settings')
        self.test_layout.addWidget(self.camera_setting_text)
        self.test_layout.addWidget(self.openCameraButton)
        #self.test_layout.addWidget(self.takePhotoButton)
        self.test_layout.addWidget(self.clearButton)
        self.test_layout.addSpacing(150)

        # add nested layout
        upper_layout.addLayout(self.canvas_layout)
        #upper_layout.addStretch()   # prevents button from stretching when window size is changed
        upper_layout.addLayout(self.test_layout)

        # add clear button to remove drawn image from canvas
        clearButton = QPushButton('Clear')
        self.canvas_layout.addWidget(clearButton)
        clearButton.clicked.connect(self.clear_image)


    def clear_image(self):
        self.canvas.fill(QtGui.QColor('white'))
        self.canvas_label.setPixmap(self.canvas)


    def camera_with_thread(self):
        self.thread = QThread(parent = self)
        self.worker = worker_camera()

        self.worker.moveToThread(self.thread)

        # connect signals and slots
        self.thread.started.connect(self.worker.worker_start_camera)
        self.worker.capture_picture.connect(self.report_capture)
        self.worker.progress.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)  

        # Start the thread
        self.thread.start()  


    def report_capture(self, input):
        if (input == 1):
            #pixmap_initial = QPixmap("images\captured_image.png")
            pixmap_initial = QPixmap('images/captured_image.png')
            pixmap_scaled = QPixmap.scaled(pixmap_initial, 400, 400)
            self.canvas_label.setPixmap(pixmap_scaled)


    def predict_show(self):
        self.prediction, self.accuracy = nnmodel.process_capture_image()
        self.prediction_browser.setText("Prediction is: " + self.prediction)
        self.prediction_browser.setFont(QFont('Serif', 10))
        self.accuracy_browser.setText("Confidence: " + self.accuracy + "%")
        self.accuracy_browser.setFont(QFont('Serif', 10))
     
    # loads from saved model
    def load_selected_model(self):
        # retrieve current text displayed in combo box to get model names
        chosen_model = self.model_combo.currentText()

        # check which model is chosen 
        if (chosen_model == 'Default_Net'):
            # load the chosen model
            nnmodel.load_model_1()
            self.selected_model_text = 'Default_Net'
            self.set_model_browser.setText("Model loaded: " + self.selected_model_text)
        elif (chosen_model == 'CNN_Net'):
            nnmodel.load_model_2()
            self.selected_model_text = 'CNN_Net'
            self.set_model_browser.setText("Model loaded: " + self.selected_model_text)
        elif (chosen_model == 'ResNet_Net'):
            nnmodel.load_model_3()
            self.selected_model_text = 'ResNet_Net'
            self.set_model_browser.setText("Model loaded: " + self.selected_model_text)
        elif (chosen_model == 'Custom_Net'):
            
            self.selected_model_text = 'Custom_Net'

            # check which model to load based on what the custom model was trained on
            model_type =  str(type(nnmodel.model).__name__)
            if (model_type == 'Default_Net' and int(nnmodel.load_model_4_Default()) != 0):
                nnmodel.load_model_4_Default()
                self.set_model_browser.setText("Model loaded: " + self.selected_model_text) 
    
            elif (model_type == 'CNN_Net' and int(nnmodel.load_model_4_Default()) != 0):
                nnmodel.load_model_4_CNN
                self.set_model_browser.setText("Model loaded: " + self.selected_model_text) 

            elif(model_type == 'ResNet_Net' and int(nnmodel.load_model_4_Default()) != 0):
                nnmodel.load_model_4_ResNet
                self.set_model_browser.setText("Model loaded: " + self.selected_model_text) 

            else:
                self.set_model_browser.setText("Model does not exist: " + self.selected_model_text) 
    

class worker_camera(QObject):
    capture_picture = pyqtSignal(int)
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def worker_start_camera(self):

        try:
            #vid = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
            vid = cv2.VideoCapture(0)
            while(1):
                ret, frame = vid.read()
                cv2.imshow('Live Camera Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif cv2.waitKey(1) & 0xFF == ord('v'):
                    #cv2.imwrite('frame.png', frame)
                    cv2.imwrite('images/captured_image.png', frame)
                    self.capture_picture.emit(1)
                    break

            vid.release()
            cv2.destroyAllWindows() 

        except AssertionError:
            print("Assertion Error: Check if camera is connected")





