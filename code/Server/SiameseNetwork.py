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

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class ModelSM:
    def __init__(self):
        self.input_shape_rgb = (480, 480, 3)
        self.input_shape_depth = (480, 480, 1)

        self.architecture()

    def architecture(self):

        rgb_branch = self.create_rgb_branch()
        depth_branch = self.create_depth_branch()

        input_rgb_a = Input(shape=self.input_shape_rgb)
        input_rgb_b = Input(shape=self.input_shape_rgb)
        input_depth_a = Input(shape=self.input_shape_depth)
        input_depth_b = Input(shape=self.input_shape_depth)

        features_rgb_a = rgb_branch(input_rgb_a)
        features_rgb_b = rgb_branch(input_rgb_b)
        features_depth_a = depth_branch(input_depth_a)
        features_depth_b = depth_branch(input_depth_b)

        combined_a = concatenate([features_rgb_a, features_depth_a])
        combined_b = concatenate([features_rgb_b, features_depth_b])

        distance = Lambda(self.euclidean_distance)([combined_a, combined_b])

        self.siamese_network = Model(
            inputs=[input_rgb_a, input_rgb_b, input_depth_a, input_depth_b],
            outputs=distance,
        )

        self.siamese_network.compile(loss=self.contrastive_loss, optimizer="adam")

    def create_rgb_branch(self):
        input_rgb = Input(shape=self.input_shape_rgb)
        x = Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape_rgb)(
            input_rgb
        )
        x = MaxPooling2D()(x)
        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        model_rgb = Model(input_rgb, x)
        return model_rgb

    def create_depth_branch(self):
        input_depth = Input(shape=self.input_shape_depth)
        y = Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape_depth)(
            input_depth
        )
        y = MaxPooling2D()(y)
        y = Conv2D(64, (3, 3), activation="relu")(y)
        y = MaxPooling2D()(y)
        y = Flatten()(y)
        y = Dense(128, activation="relu")(y)
        model_depth = Model(input_depth, y)
        return model_depth

    def euclidean_distance(self,vectors):
        (featuresA, featuresB) = vectors
        sum_squared = K.sum(K.square(featuresA - featuresB), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_squared, K.epsilon()))

    def contrastive_loss(self,y_true, y_pred):
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def load_and_preprocess_image(self, image_path, size=(480, 480), grayscale=False):
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(image_path, flag)
        if img is None:
            raise ValueError(f"Error al cargar la imagen: {image_path}")
        img = cv2.resize(img, size)
        img = img.reshape((size[0], size[1], 1)) if grayscale else img
        return img / 255.0

    def training(self):
        csv_file = "../data/data.csv"
        data = pd.read_csv(csv_file)

        faces1_rgb, faces2_rgb = [], []
        faces1_depth, faces2_depth = [], []
        labels = []

        for idx, row in data.iterrows():
            face1_rgb = self.load_and_preprocess_image(row["Face1"])
            face2_rgb = self.load_and_preprocess_image(row["Face2"])
            depth1 = self.load_and_preprocess_image(
                row["Depth1"], size=(480, 480), grayscale=True
            )
            depth2 = self.load_and_preprocess_image(
                row["Depth2"], size=(480, 480), grayscale=True
            )

            faces1_rgb.append(face1_rgb)
            faces2_rgb.append(face2_rgb)
            faces1_depth.append(depth1)
            faces2_depth.append(depth2)
            labels.append(row["Validation"])

        faces1_rgb = np.array(faces1_rgb)
        faces2_rgb = np.array(faces2_rgb)
        faces1_depth = np.array(faces1_depth)
        faces2_depth = np.array(faces2_depth)
        labels = np.array(labels)

        self.siamese_network.fit(
            [faces1_rgb, faces2_rgb, faces1_depth, faces2_depth],
            labels,
            batch_size=32,
            epochs=20,
        )

    def save_weights(self):
        self.siamese_network.save_weights(
            "../../data/siamese_network.weights.h5", overwrite=True
        )

    def load_weights(self):
        self.siamese_network.load_weights(filepath="../../data/siamese_network.weights.h5")

    def predict_similarity(self,img_rgb_a, img_depth_a, img_rgb_b, img_depth_b):
        img_rgb_a = np.expand_dims(self.load_and_preprocess_image(img_rgb_a), axis=0)
        img_depth_a = np.expand_dims(
            self.load_and_preprocess_image(img_depth_a, grayscale=True), axis=0
        )
        img_rgb_b = np.expand_dims(self.load_and_preprocess_image(img_rgb_b), axis=0)
        img_depth_b = np.expand_dims(
            self.load_and_preprocess_image(img_depth_b, grayscale=True), axis=0
        )
        distance = self.siamese_network(
            [img_rgb_a, img_rgb_b, img_depth_a, img_depth_b]
        )
        return distance
