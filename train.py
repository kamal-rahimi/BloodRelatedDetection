"""
Creates a convolutionaly neural network (CNN) in Keras and trains the network
to identify if two pictures belong to people who are blood related.
"""

import tensorflow as tf
from tensorflow.python import keras
from keras import Model
from keras import backend as K
from keras.models import load_model, save_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

from cvision_tools import detect_face, crop_face, convert_to_gray, resize_with_pad, over_sample, under_sample

from FamilyInWildDataset import FamilyInWildDataset

import numpy as np
import pickle
import os
import cv2

import matplotlib.pyplot as plt


image_height = 64
image_width = 64
image_n_channels = 1

n_epochs = 100
batch_size = 10

RELATION_PREDICTION_MODEL_PATH = "./model/relation_model"

FIW_DATA_FILE_PATH = "./data/family_in_wild.pickle"


def prepare_relation_data():
    """ Prepares training and test data for emotion detection model from Family in Wild (FIW)
    image dataset 
    Args:
    Returns:
        X_train: a numpy array of the train face image data
        X_test: a numpy array of the test face image data
        y_emotion_train: a numpy array of the train emotion lables
        y_emotion_test: a numpy array of the test emotion lables
    """
    dataset = FamilyInWildDataset()
    dataset.read(image_height=image_height, image_width=image_width)

    X_train = np.array(dataset.X_train).astype('float32')
    X_test  = np.array(dataset.X_test).astype('float32')
    X_train = X_train / 128 - 1
    X_test  = X_test  / 128 - 1


    y_emotion_train = np.array(dataset.y_train)
    y_emotion_test  = np.array(dataset.y_test)

    return X_train, X_test, y_emotion_train, y_emotion_test

def create_relation_encoder_model(X_train, X_test):
    """ Creates a convoluational neural network (CNN) and trains the model to detect facial
     emotion in an input image
    Args:
        X_train: a numpy array of the train image data
        X_test: a numpy array of the test image data
        y_train: a numpy array of the train emotion lables
        y_test: a numpy array of the test emotion lables
    Returns:
    """
    input_image = Input(shape=(image_height, image_width, image_n_channels))
    x = Conv2D(8,(3,3), activation='elu', padding='same')(input_image)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(16,(3,3), activation='elu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(16,(3,3), activation='elu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(16,(3,3), activation='elu', padding='same')(x)
    encoded = MaxPooling2D((2,2), padding='same')(x)

    x = Conv2D(16, (3, 3), activation='elu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='tanh', padding='same')(x)
    
    autoencoder = Model(input_image, decoded)
    autoencoder.compile(optimizer='nadam', loss='mse', metrics=[])
    autoencoder.summary()
    autoencoder.fit(X_train, X_train, epochs=n_epochs, batch_size=batch_size, shuffle=True)

    return autoencoder


def train_relation_model(autoencoder, X_train, X_test):
    """ Creates a convoluational neural network (CNN) and trains the model to detect facial
     emotion in an input image
    Args:
        X_train: a numpy array of the train image data
        X_test: a numpy array of the test image data
        y_train: a numpy array of the train emotion lables
        y_test: a numpy array of the test emotion lables
    Returns:
    """
    pass




def main():

    if os.path.isfile(FIW_DATA_FILE_PATH):
        X_train, X_test, y_train, y_test = pickle.load(open(FIW_DATA_FILE_PATH, 'rb'))
    else:
        X_train, X_test, y_train, y_test = prepare_relation_data()
        #X_train, y_train = over_sample(X_train, y_train)
        #X_test, y_test = over_sample(X_test, y_test)
        pickle.dump([X_train, X_test, y_train, y_test], open(FIW_DATA_FILE_PATH, 'wb'))
    
    print("Train size: {}".format(len(X_train)))
    print("Test size: {}" .format(len(X_test)))

    if os.path.isfile(RELATION_PREDICTION_MODEL_PATH):
        autoencoder = load_model(RELATION_PREDICTION_MODEL_PATH)
    else:
        autoencoder = create_relation_encoder_model(X_train, X_train)
        autoencoder.save(RELATION_PREDICTION_MODEL_PATH)
    
    encoder = K.function([autoencoder.layers[0].input], [autoencoder.layers[7].output])

    images = X_train[:10]
    images_rec = autoencoder.predict(images)
    images += 1.0
    images_rec += 1.0

    images *= 127.0
    images_rec *= 127.0
    images = np.floor(images).astype(np.uint8)
    images_rec = np.floor(images_rec).astype(np.uint8)
    for count in range(10):
        #plt.imshow(images[count].squeeze())
        #plt.show()
        cv2.imshow("image", images[count])
        cv2.waitKey(0)
        #plt.imshow(images_rec[count].squeeze())
        #plt.show()
        cv2.imshow("image_rec", np.array(images_rec[count]))
        cv2.waitKey(0)


    #train_relation_model(autoencoder, X_train, X_test, y_train, y_test)
    


if __name__ == "__main__":
    main()