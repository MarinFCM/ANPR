from keras.layers import *
from keras.models import Sequential
from keras.losses import *
from keras.optimizers import *
import tensorflow as tf
import pandas as pd
import CustomCallback
from sklearn.preprocessing import LabelBinarizer
import cv2
import numpy as np
import os


class ANPRModel:

    def read_image(self, image_path, label):
        image = tf.io.read_file(self.trainDirectory + image_path)
        image = tf.image.decode_png(image, channels=1, dtype=tf.uint16)
        print(label)
        return image, label

    def read_test_image(self, image_path, label):
        image = tf.io.read_file(self.validDirectory + image_path)
        image = tf.image.decode_png(image, channels=1, dtype=tf.uint16)
        print(label)
        return image, label

    def preprocess(self, image, label):
        resize_height = 50
        resize_width = 20
        image = tf.image.resize(image, (resize_height, resize_width))
        image /= 255.0
        # image = tf.image.rgb_to_grayscale(image)
        # image = tf.image.random_brightness(image, max_delta=0.1)
        # image = tf.image.random_contrast(image, lower=0.1, upper=0.2)

        return image, label

    def pack_features_vector(self, features, labels):
        features = tf.reshape(features, (50, 20, 1))
        return features, labels

    def __init__(self, gui):
        self.gui = gui

    def trainModel(self, trainPath, trainLabelPath, validPath, validLabelPath, trainOnGPU, noOfEpochs=50):

        if trainOnGPU == 1:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)

        self.trainDirectory = trainPath + '/'
        self.trainData = pd.read_csv(trainLabelPath)
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                            'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Z']
        file_paths = self.trainData['file_name'].values
        labels = self.trainData['label'].values
        labelsInt = []
        for label in labels:
            labelsInt.append(self.class_names.index(label))
        self.ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labelsInt))
        self.ds_train = self.ds_train.map(self.read_image).map(self.preprocess).map(self.pack_features_vector).batch(1)

        self.validDirectory = validPath + '/'
        self.validData = pd.read_csv(validLabelPath)

        file_paths_test = self.validData['file_name'].values
        labelsTest = self.validData['label'].values
        labelsIntTest = []
        for label in labelsTest:
            labelsIntTest.append(self.class_names.index(label))
        self.ds_test = tf.data.Dataset.from_tensor_slices((file_paths_test, labelsIntTest))
        self.ds_test = self.ds_test.map(self.read_test_image).map(self.preprocess).map(self.pack_features_vector).batch(1)

        self.model = Sequential(
            [
                Input((50, 20, 1)),
                Conv2D(128, (4, 4), padding="same", activation="relu"),
                MaxPooling2D(),
                Conv2D(64, (4, 4), padding="same", activation="relu"),
                MaxPooling2D(),
                Conv2D(32, (4, 4), padding="same", activation="relu"),
                MaxPooling2D(),
                Flatten(),
                Dense(64, activation="relu"),
                Dense(len(self.class_names)),
            ]
        )

        self.model.compile(optimizer=Adam(learning_rate=0.0004),
                      loss=[
                          SparseCategoricalCrossentropy(from_logits=True),
                      ],
                      metrics=["accuracy"],
                      )
        cb = tf.keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)
        if trainOnGPU == 1:
            with tf.device('/device:GPU:0'):
                self.model.fit(self.ds_train, validation_data=self.ds_test, epochs=noOfEpochs, verbose=2, callbacks=[CustomCallback.CustomCallback(self.gui, noOfEpochs)])
        else:
            with tf.device('/device:CPU:0'):
                self.model.fit(self.ds_train, validation_data=self.ds_test, epochs=noOfEpochs, verbose=2, callbacks=[CustomCallback.CustomCallback(self.gui, noOfEpochs)])
        self.gui.updateProgressbar(100)
        self.gui.Status.configure(text='''TRAINING DONE''')
        self.model.save('my_model')

    def predictImage(self):
        pass
