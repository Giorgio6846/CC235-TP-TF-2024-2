import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    concatenate,
    Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import cv2

print(tf.config.list_physical_devices('GPU'))

def create_rgb_branch(input_shape):
    input_rgb = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu", input_shape=input_shape)(input_rgb)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    model_rgb = Model(input_rgb, x)
    return model_rgb


def create_depth_branch(input_shape):
    input_depth = Input(shape=input_shape)
    y = Conv2D(32, (3, 3), activation="relu", input_shape=input_shape)(input_depth)
    y = MaxPooling2D()(y)
    y = Conv2D(64, (3, 3), activation="relu")(y)
    y = MaxPooling2D()(y)
    y = Flatten()(y)
    y = Dense(128, activation="relu")(y)
    model_depth = Model(input_depth, y)
    return model_depth


# Función para calcular la distancia euclidiana entre las salidas
def euclidean_distance(vectors):
    (featuresA, featuresB) = vectors
    sum_squared = K.sum(K.square(featuresA - featuresB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

input_shape_rgb = (480, 480, 3)
input_shape_depth = (480, 480, 1)

rgb_branch = create_rgb_branch(input_shape_rgb)
depth_branch = create_depth_branch(input_shape_depth)

input_rgb_a = Input(shape=input_shape_rgb)
input_rgb_b = Input(shape=input_shape_rgb)
input_depth_a = Input(shape=input_shape_depth)
input_depth_b = Input(shape=input_shape_depth)

features_rgb_a = rgb_branch(input_rgb_a)
features_rgb_b = rgb_branch(input_rgb_b)
features_depth_a = depth_branch(input_depth_a)
features_depth_b = depth_branch(input_depth_b)

combined_a = concatenate([features_rgb_a, features_depth_a])
combined_b = concatenate([features_rgb_b, features_depth_b])

distance = Lambda(euclidean_distance)([combined_a, combined_b])

siamese_network = Model(
    inputs=[input_rgb_a, input_rgb_b, input_depth_a, input_depth_b], outputs=distance
)

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

siamese_network.compile(loss=contrastive_loss, optimizer="adam")

def load_and_preprocess_image(image_path, size=(480, 480), grayscale=False):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(image_path, flag)
    if img is None:
        raise ValueError(f"Error al cargar la imagen: {image_path}")
    img = cv2.resize(img, size)
    img = img.reshape((size[0], size[1], 1)) if grayscale else img
    return img / 255.0

# Cargar el archivo .csv
csv_file = "../data/data.csv"
data = pd.read_csv(csv_file)

faces1_rgb, faces2_rgb = [], []
faces1_depth, faces2_depth = [], []
labels = []

for idx, row in data.iterrows():
    face1_rgb = load_and_preprocess_image(row["Face1"])
    face2_rgb = load_and_preprocess_image(row["Face2"])
    depth1 = load_and_preprocess_image(row["Depth1"], size=(480, 480), grayscale=True)
    depth2 = load_and_preprocess_image(row["Depth2"], size=(480, 480), grayscale=True)

    faces1_rgb.append(face1_rgb)
    faces2_rgb.append(face2_rgb)
    faces1_depth.append(depth1)
    faces2_depth.append(depth2)
    labels.append(
        row["Validation"]
    ) 

faces1_rgb = np.array(faces1_rgb)
faces2_rgb = np.array(faces2_rgb)
faces1_depth = np.array(faces1_depth)
faces2_depth = np.array(faces2_depth)
labels = np.array(labels)

siamese_network.fit(
    [faces1_rgb, faces2_rgb, faces1_depth, faces2_depth],
    labels,
    batch_size=32,
    epochs=20,
)

distances = siamese_network.predict(
    [faces1_rgb, faces2_rgb, faces1_depth, faces2_depth]
)
print("Distancias predichas:", distances)
