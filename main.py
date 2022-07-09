# File name: main.py
# Author: Francis Cho
# Project Version: 2.0
# Description: Main script that executes all the files in scripts folder     
# Python Version: 3.1

import sys
from PyQt5.QtWidgets import QApplication

from scripts.myGUI import myGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = myGUI()
    sys.exit(app.exec_())    
