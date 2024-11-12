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
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtCore import *
import requests
from io import BytesIO
from PIL import Image

class UserCreation(QWidget):
    def updateImages(self):
        self.dataApp.Images = self.networkController.getImages(self.dataApp.getIPAddress())

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def showImages(self):
        self.clear_layout(self.imageLayout)

        if len(self.dataApp.Images) == 0:
            print("Images not found")
            return None

        for image_data in self.dataApp.Images:
            print(image_data)

            pixmap = QPixmap()
            image = Image.open(BytesIO(image_data))
            img_byte_array = BytesIO()
            image.save(img_byte_array, format='PNG' if image.format == 'PNG' else 'JPEG')

            pixmap.loadFromData(img_byte_array.getvalue())
            scaled_pixmap = pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)

            label = QLabel()
            label.setPixmap(scaled_pixmap)
            self.imageLayout.addWidget(label)

    def responseToServer(self):
        response = self.networkController.createUser(
            self.dataApp.getIPAddress(), 
            self.dataApp.getUsername()
        )
        print(response)

    def __init__(self, data, networkController):
        super().__init__()

        self.dataApp = data
        self.networkController = networkController

        self.IPAddressDialog = QInputDialog()
        self.IPAddressDialog.setLabelText("Direccion de IP")
        self.IPAddressDialog.setInputMode(QInputDialog.TextInput)
        self.IPAddressDialog.setOption(QInputDialog.NoButtons, True)

        self.refreshDataButton = QPushButton()
        self.refreshDataButton.setText("Obtain from server")

        self.showImagesButton = QPushButton()
        self.showImagesButton.setText("Show Images")

        self.IPAddressDialog.textValueChanged.connect(self.dataApp.setIPAddress)
        self.refreshDataButton.clicked.connect(self.updateImages)
        self.showImagesButton.clicked.connect(self.showImages)

        self.LayoutIP = QHBoxLayout()
        self.LayoutIP.addWidget(self.IPAddressDialog)
        self.LayoutIP.addWidget(self.refreshDataButton)
        self.LayoutIP.addWidget(self.showImagesButton)

        self.userCreationLayout = QHBoxLayout()
        self.usernameDialog = QInputDialog()
        self.usernameDialog.setLabelText("Nombre del usuario")
        self.usernameDialog.setInputMode(QInputDialog.TextInput)
        self.usernameDialog.setOption(QInputDialog.NoButtons, True)

        self.sendResponseButton = QPushButton()
        self.sendResponseButton.setText("Send to Server")

        self.usernameDialog.textValueChanged.connect(self.dataApp.setUsername)
        self.sendResponseButton.clicked.connect(self.responseToServer)

        self.userCreationLayout.addWidget(self.usernameDialog)
        self.userCreationLayout.addWidget(self.sendResponseButton)

        self.imageLayout = QHBoxLayout()

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.LayoutIP)
        self.layout.addLayout(self.userCreationLayout)
        self.layout.addLayout(self.imageLayout)
        self.setLayout(self.layout)
