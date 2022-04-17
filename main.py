import sys
from PyQt5.QtWidgets import QApplication

from scripts.myGUI import myGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = myGUI()
    sys.exit(app.exec_())    
