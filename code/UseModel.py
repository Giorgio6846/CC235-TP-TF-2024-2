from keras import saving
import pandas as pd
import cv2
import numpy as np
from Server.SiameseNetwork import ModelSM
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    concatenate,
    Lambda,
)
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

siamese_network = ModelSM()

siamese_network.load_weights()

print("PRUEBA: MISMA PERSONA")

# Ejemplo de inferencia
distance = siamese_network.predict_similarity(
    "../data/Dataset_2/User1/Test/FaceImage20241112204422.jpg",
    "../data/Dataset_2/User1/Test/DepthImage20241112204422.png",
    "../data/Dataset_2/User1/Validation/FaceImage20241112204140.jpg",
    "../data/Dataset_2/User1/Validation/DepthImage20241112204140.png",
)
if distance < 0.5:  # Define un umbral adecuado
    print("Misma persona")
else:
    print("Persona diferente")

print("PRUEBA: DIFERENTE PERSONA")
print(distance)


# Ejemplo de inferencia
distance = siamese_network.predict_similarity(
    "../data/NU_Dataset_1/Images/FaceNorm/80TD_RGB_A_3_4.jpg",
    "../data/NU_Dataset_1/Images/DepthNorm/80TD_RGB_A_3_4.jpg",
    "../data/Dataset_2/User1/Validation/FaceImage20241112204140.jpg",
    "../data/Dataset_2/User1/Validation/DepthImage20241112204140.png",
)
if distance < 0.5:  # Define un umbral adecuado
    print("Misma persona")
else:
    print("Persona diferente")
print(distance)
