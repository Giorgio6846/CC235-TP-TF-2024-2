import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QStackedLayout,
    QPushButton,
)
from enum import Enum
from Screens import InitScreen, UserCreation
import requests

Options = Enum("Options", ["MainScreen", "ExitProgram", "UserCreation", "UserVerification"])

class Data():
    IPAddress = ""
    Images = []

    def __init__(self):
        self.IPAddress = ""
        self.username = ""

    def setIPAddress(self, IP):
        self.IPAddress = IP
        print(self.IPAddress)

    def getIPAddress(self):
        return self.IPAddress

    def setUsername(self, username):
        self.username = username
        print(self.username)

    def getUsername(self):
        return self.username

class NetworkController():
    def __init__(self):
        pass

    def getImages(self,IP):
        print("IP Address: " + f"{IP}")

        url = f"http://{IP if IP else '10.0.1.12:5050'}/face-depth-data"
        response = requests.get(url)

        content_type = response.headers.get("Content-Type")
        boundary = content_type.split("boundary=")[-1]

        # Split the response content by the boundary
        parts = response.content.split(f"--{boundary}".encode())
        images = []

        for part in parts:
            if b"Content-Type: image/" in part:
                image_data = part.split(b"\r\n\r\n", 1)[-1].rsplit(b"\r\n", 1)[0]
                images.append(image_data)

        return images

    def createUser(self, IP, username):
        if username == "":
            return -1

        url = f"http://{IP if IP else '10.0.1.12:5050'}/face-creation"
        response = requests.post(
            url,
            json={"username": f"{username}"},
            headers={"Content-Type": "application/json"},
        )

        print(response.json())

        if response.status_code == 200:
            return True
        else:
            return False

class Controller(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.dataApp = Data()
        self.networkController = NetworkController()

        self.index = 0
        self.LayoutButton = QHBoxLayout()

        self.buttonStart = QPushButton(text="Menu")
        self.buttonUserCreation = QPushButton(text="Creacion de Usuario")
        self.buttonUserView = QPushButton(text="Verificacion de Usaurio")
        self.buttonExit = QPushButton(text="Salir")

        self.buttonStart.clicked.connect(lambda: self.setIndex(Options.MainScreen))
        self.buttonUserCreation.clicked.connect(lambda: self.setIndex(Options.UserCreation))
        # self.buttonResult.clicked.connect(lambda: self.setIndex(Options.ResultMatrix))
        self.buttonExit.clicked.connect(lambda: self.setIndex(Options.ExitProgram))

        self.LayoutButton.addWidget(self.buttonStart)
        self.LayoutButton.addWidget(self.buttonUserCreation)
        self.LayoutButton.addWidget(self.buttonExit)

        self.layout = QStackedLayout()

        self.mainScreen = InitScreen.InitScreen(self.dataApp, self.networkController)
        self.userCreation = UserCreation.UserCreation(self.dataApp, self.networkController)

        self.layout.addWidget(self.mainScreen)
        self.layout.addWidget(self.userCreation)

        self.layout.setCurrentIndex(self.index)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.addLayout(self.LayoutButton)
        self.mainLayout.addLayout(self.layout)
        self.setLayout(self.mainLayout)

    def initUI(self):
        self.setFixedSize(800, 600)
        self.setWindowTitle("Proyecto - Procesamiento de Imagenes 2024-2")

    def setIndex(self, caseButton):
        if caseButton == Options.ExitProgram:
            sys.exit(app.exec())
        elif caseButton == Options.MainScreen:
            self.index = 0
        elif caseButton == Options.UserCreation:
            self.index = 1
        elif caseButton == Options.UserVerification:
            self.index = 2

        self.layout.setCurrentIndex(self.index)

if __name__ == "__main__":
    app = QApplication([])
    mainWindow = Controller()
    mainWindow.show()

    sys.exit(app.exec())
