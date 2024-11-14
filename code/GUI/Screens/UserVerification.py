from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QInputDialog,
    QComboBox
)
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import *
from io import BytesIO
from PIL import Image
import json

class UserVerification(QWidget):
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

        response = self.networkController.verifyUser(
            self.dataApp.getIPAddress(), self.usersList.currentText()
        )

        percentageClassic = response.get("PercentageClassic") * 100
        percentageDL = response.get("PercentageDL") * 100

        print(percentageClassic, percentageDL)

        self.classicalPercentage.setText(f"{percentageClassic:.2f}%")
        self.DeepLearningPercentage.setText(f"{percentageDL:.2f}%")

    def refreshData(self):
        self.dataApp.Images = self.networkController.getImages(
            self.dataApp.getIPAddress()
        )

        self.dataApp.userList = self.networkController.getUsers(
            self.dataApp.getIPAddress()
        )

        self.fillComboBox()

    def fillComboBox(self):
        # users = json.loads()
        self.usersList.clear()

        items = list(self.dataApp.userList.values())

        print(items)
        self.usersList.addItems(items)

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
        self.refreshDataButton.clicked.connect(self.refreshData)
        self.showImagesButton.clicked.connect(self.showImages)

        self.LayoutIP = QHBoxLayout()
        self.LayoutIP.addWidget(self.IPAddressDialog)
        self.LayoutIP.addWidget(self.refreshDataButton)
        self.LayoutIP.addWidget(self.showImagesButton)

        self.sendResponseButton = QPushButton()
        self.sendResponseButton.setText("Send to Server")

        self.usersList = QComboBox()
        self.usersLabel = QLabel()
        self.usersLabel.setText("Lista de usuarios")      

        self.sendResponseButton = QPushButton()
        self.sendResponseButton.setText("Send to Server")

        self.sendResponseButton.clicked.connect(self.responseToServer)

        self.users = QVBoxLayout()
        self.users.addWidget(self.usersLabel)
        self.users.addWidget(self.usersList)

        self.userUpdate = QHBoxLayout()
        self.userUpdate.addLayout(self.users)
        self.userUpdate.addWidget(self.sendResponseButton)

        self.imageLayout = QHBoxLayout()

        self.percentagesBox = QHBoxLayout()

        self.classicBox = QVBoxLayout()
        self.classicalHeader = QLabel()
        self.classicalPercentage = QLabel()

        self.classicalHeader.setText("Classsic Method")
        self.classicalHeader.setFont(QFont("Arial", 18))
        self.classicalPercentage.setText("%")
        self.classicalPercentage.setFont(QFont("Arial", 18))
        self.classicBox.addWidget(self.classicalHeader)
        self.classicBox.addWidget(self.classicalPercentage)

        self.DeepLearningBox = QVBoxLayout()
        self.DeepLearningHeader= QLabel()
        self.DeepLearningPercentage = QLabel()

        self.DeepLearningHeader.setText("Deep Learning Method")
        self.DeepLearningHeader.setFont(QFont("Arial", 18))
        self.DeepLearningPercentage.setText("%")
        self.DeepLearningPercentage.setFont(QFont("Arial", 18))
        self.DeepLearningBox.addWidget(self.DeepLearningHeader)
        self.DeepLearningBox.addWidget(self.DeepLearningPercentage)

        self.percentagesBox.addLayout(self.classicBox)
        self.percentagesBox.addLayout(self.DeepLearningBox)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.LayoutIP)
        self.layout.addLayout(self.userUpdate)
        self.layout.addLayout(self.imageLayout)
        self.layout.addLayout(self.percentagesBox)
        self.setLayout(self.layout)
