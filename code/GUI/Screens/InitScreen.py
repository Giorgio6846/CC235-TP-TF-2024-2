from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PySide6.QtGui import QFont
from PySide6.QtCore import *

class InitScreen(QWidget):

    def __init__(self, data, networkController):
        super().__init__()
        
        self.dataApp = data
        self.networkController = networkController
        
        self.TitleScreen = QLabel("Proyecto Procesamiento de Imagenes")
        self.TitleScreen.setFont(QFont("Arial", 24))
        self.TitleScreen.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.TitleScreen.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.TitleScreen.setFixedHeight(30)

        self.Integrantes = QLabel(
            
            "INTEGRANTES \n" +
            "Fabio Osorio Ramos              202211499 \n"+
            "Giorgio Mancusi Barreda       202216613\n"+
            "Mathias Hualtibamba Valerio 202214421 \n"
        )
        self.Integrantes.setFont(QFont("Arial", 24))
        self.Integrantes.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.TitleScreen)
        self.layout.addWidget(self.Integrantes)
        self.setLayout(self.layout)
