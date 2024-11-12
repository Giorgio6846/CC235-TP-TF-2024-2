from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QSizePolicy,
    QHBoxLayout,
    QPushButton,
    QInputDialog,
    QFormLayout,
)
from PySide6.QtGui import QFont
from PySide6.QtCore import *
import requests

class UserVerification(QWidget):
    def __init__(self, data):
        super().__init__()
        self.dataApp = data

    def serverResponse(self):
        pass

    def serverRequest(self):
        pass