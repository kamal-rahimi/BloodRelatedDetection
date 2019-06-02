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

import pandas as pd

import matplotlib.pyplot as plt


image_height = 64
image_width = 64
image_n_channels = 1

n_epochs = 100
batch_size = 10

RELATION_PREDICTION_MODEL_PATH = "./model/relation_model"

FIW_DATA_FILE_PATH = "./data/family_in_wild.pickle"



def prepare_data(X_train, X_test):
    """ Prepares training and test data for emotion detection model from Family in Wild (FIW)
    image dataset 
    Args:
        dataset: an object of the FamilyInWildDataset class
    Returns:
        X_train: a numpy array of the train face image data
        X_test: a numpy array of the test face image data
    """    
    X_train = np.array(X_train).astype('float32')
    X_test  = np.array(X_test).astype('float32')
    X_train = X_train / 128 - 1
    X_test  = X_test  / 128 - 1

    return X_train, X_test

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

def prepare_relation_info(relation_dict, y):
    """ Prepares training and test data for emotion detection model from Family in Wild (FIW)
    image dataset 
    Args:
        dataset: an object of the FamilyInWildDataset class
    Returns:
        X_related_dict: a Python dictionary of image indices related to an index
        X_not_related_dict: a Python dictionary of image indices not related to an index
    """
    num_indices = len(y)

    X_related_dict = {}
    X_not_related_dict = {}
    for idx1 in range(num_indices):
        X_related_dict[idx1] = []
        X_not_related_dict[idx1] = []
        for idx2 in range(max(0, idx1-100), min(num_indices, idx1+100)):
            if (y[idx2] == y[idx1]) or ((y[idx1][0] in relation_dict) and (y[idx2][0] in relation_dict[y[idx1][0]]) ):
                X_related_dict[idx1].append(idx2)
        for idx2 in range(max(0, idx1-200), min(num_indices, idx1+200)):
            if (y[idx2] != y[idx1]) and ( (y[idx1][0] not in relation_dict) or (y[idx2][0] not in relation_dict[y[idx1][0]]) ):
                X_not_related_dict[idx1].append(idx2)
                if ( len(X_not_related_dict[idx1]) == len(X_related_dict[idx1]) ):
                    break;

    return X_related_dict, X_not_related_dict

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
        dataset = pickle.load(open(FIW_DATA_FILE_PATH, 'rb'))
    else:
        dataset = FamilyInWildDataset()
        dataset.read(image_height=image_height, image_width=image_width)
        dataset.read_relations()
        pickle.dump(dataset, open(FIW_DATA_FILE_PATH, 'wb'))
    
    X_train, X_test = prepare_data(dataset.X_train, dataset.X_test)
    X_train_related_dict, X_train_not_related_dict = prepare_relation_info(dataset.relation_dict, dataset.y_train)
    X_test_related_dict,  X_test_not_related_dict  = prepare_relation_info(dataset.relation_dict, dataset.y_test)

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
    """
    for count in range(10):
        #plt.imshow(images[count].squeeze())
        #plt.show()
        cv2.imshow("image", images[count])
        cv2.waitKey(0)
        #plt.imshow(images_rec[count].squeeze())
        #plt.show()
        cv2.imshow("image_rec", np.array(images_rec[count]))
        cv2.waitKey(0)

    """
    #train_relation_model(autoencoder, X_train, X_test, y_train, y_test)
    


if __name__ == "__main__":
    main()