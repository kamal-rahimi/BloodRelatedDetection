"""
Creates a convolutional neural network (CNN) in Keras and trains the network
to identify if two pictures belong to people who are blood related.
"""

import tensorflow as tf
from tensorflow.python import keras
from keras import Model
from keras import backend as K
from keras.models import load_model, save_model
from keras.layers import Input, Dense, Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Dropout, Reshape, Flatten, Concatenate, Add, Subtract, Lambda, regularizers

from cvision_tools import detect_face, crop_face, convert_to_gray, resize_with_pad, over_sample, under_sample

from FamilyInWildDataset import FamilyInWildDataset

import numpy as np
import pickle
import os
import cv2
import random

import pandas as pd

import matplotlib.pyplot as plt


image_height = 64
image_width = 64
image_n_channels = 1

n_epochs = 20
batch_size = 10

RELATION_DETECTION_EMBEDDING_MODEL_PATH = "./model/blood_related_detect_embedding"
RELATION_DETECTION_CNN_MODEL_PATH = "./model/blood_related_detect_model"

FIW_DATA_FILE_PATH = "./data/family_in_wild.pickle"

num_train_samples = 150000
num_test_samples = 24000

def read_dataset():
    """ Reads train and test data from Familyin Wild (FIW) dataset
    Args:
    Returns:
        dataset: an object of class FamilyInWildDataset
    """
    if os.path.isfile(FIW_DATA_FILE_PATH):
        dataset = pickle.load(open(FIW_DATA_FILE_PATH, 'rb'))
    else:
        dataset = FamilyInWildDataset()
        dataset.read(image_height=image_height, image_width=image_width, gray=True)
        dataset.read_relations()
        pickle.dump(dataset, open(FIW_DATA_FILE_PATH, 'wb'))
    return dataset


def prepare_data(X_train, X_test):
    """ Prepares training and test data for the model from Family in Wild (FIW) image dataset 
    Args:
        X_train: a numpy array of the train image data
        X_test: a numpy array of the test image data
    Returns:
        X_train: a numpy array of the train image data
        X_test: a numpy array of the test image data
    """    
    X_train = np.array(X_train).astype('float32')
    X_test  = np.array(X_test).astype('float32')
    X_train = X_train / 128 - 1
    X_test  = X_test  / 128 - 1

    return X_train, X_test

def create_relation_dictionaris(X, y, relation_dict):
    """ Creates a dictionary containing the ID of images that are blood related to an image ID
    Args:
        X: numpy array of data
        y: numpy array of labels
        relation_dict: a dictionary containing people who are blood related
    Returns:
        X_related_dict: a dictionary containing the ID of images that are blood related to the image
    """
    num_indices = len(y)

    X_related_dict = {}
    num_related = 0
    for idx1 in range(num_indices):
        X_related_dict[idx1] = []
        for idx2 in range(num_indices):
            if ( (y[idx2] == y[idx1]) and (idx1 != idx2)) or ((y[idx1][0] in relation_dict) and (y[idx2][0] in relation_dict[y[idx1][0]]) ):
                X_related_dict[idx1].append(idx2)
                num_related += 1
                
    print("Data size:", num_related )
    
    return X_related_dict

def data_generator(X, y, data_len, relation_dict, batch_size):
    """ Creates a datagenrator to feed two images and label (related or not related) to the
    relation detection model
    Args:
        X: input images
        y: input image lables
        relation_dict: a python dictionary containing relation information
        return_index: weather return indices of the two image along with data or not
    Returns:
         yields with numpy arrays contining two pair of images and lable (related, not related)
    """
    X_related_dict = create_relation_dictionaris(X, y, relation_dict)
    X_clf = []
    y_clf = []
    while True:      
        for idx1 in range(len(X)):
            for idx2 in X_related_dict[idx1]:
                X_clf.append([ X[idx1], X[idx2] ] )
                y_clf.append([1])
                if ( len(y_clf) == batch_size ):
                    yield(np.array(X_clf), np.array(y_clf))
                    X_clf = []
                    y_clf = []

            random_indecies = list(np.random.randint(len(X), size=20))
            random_indecies_in = 0
            for i in range(len(X_related_dict[idx1])):
                while y[random_indecies[random_indecies_in]] == y[idx1]  or random_indecies[random_indecies_in] in X_related_dict[idx1]:
                    random_indecies_in += 1
                idx3 = random_indecies[random_indecies_in]

                X_clf.append([ X[idx1], X[idx3] ] )
                y_clf.append([0])
                if ( len(y_clf) == batch_size ):
                    yield(np.array(X_clf), np.array(y_clf))
                    X_clf = []
                    y_clf = []

def embedding_data_generator(X, y, data_len, relation_dict, batch_size):
    """ Creates a datagenrator to feed three images to train embedding model.
     first image is the anchor, second image is related to the anchor and third image is
     unrelated to the anchor image
    Args:
        X: input images
        y: input image lables
        relation_dict: a python dictionary containing relation information
    Returns:
         yields with numpy arrays contining thjree images
    """
    X_related_dict = create_relation_dictionaris(X, y, relation_dict)
    X_clf = []
    y_clf = []
    num_data = 0
    while True:  
        for idx1 in range(len(X)):
            random_indecies = list(np.random.randint(len(X), size=20))
            random_indecies_in = 0
            for idx2 in X_related_dict[idx1]:
                while y[random_indecies[random_indecies_in]] == y[idx1] or random_indecies[random_indecies_in] in X_related_dict[idx1]:
                    random_indecies_in += 1
                idx3 = random_indecies[random_indecies_in]
                X_clf.append([ X[idx1], X[idx2], X[idx3] ] )
                y_clf.append([-1])
                num_data += 1
                if ( len(y_clf) == batch_size ):
                    yield(np.array(X_clf), np.array(y_clf))
                    X_clf = []
                    y_clf = []

def emedding_model():
    """ Creates a convolutional neural network embedding model to extract a feature
    vector for a face image
    Args:
    Returns:
         embedding_model: A convolutional neural network embedding model
    """
    embedding_input = Input(shape=(image_height, image_width, image_n_channels))
    xr = Conv2D(16,(4,4), strides=(4, 4), activation='elu', padding='same' ) (embedding_input)
    x = Conv2D(8,(2,2), activation='elu', padding='same') (embedding_input)
    x = MaxPooling2D((2,2), padding='same') (x)
    x = Conv2D(16,(2,2), activation='elu', padding='same') (x)
    x = MaxPooling2D((2,2), padding='same') (x)
    x = Add()([x, xr])
    
    xr = Conv2D(64,(4,4), strides=(4, 4), activation='elu') (x)
    x = Conv2D(32,(2,2), activation='elu', padding='same') (x)
    x = MaxPooling2D((2,2), padding='same') (x)
    x = Conv2D(64,(2,2), activation='elu', padding='same') (x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Add()([x, xr])

    xr = Conv2D(128,(4,4), strides=(4, 4), activation='elu', padding='same') (x)
    x = Conv2D(96,(2,2), activation='elu', padding='same') (x)
    x = MaxPooling2D((2,2), padding='same') (x)
    x = Conv2D(128,(2,2), activation='elu', padding='same') (x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Add()([x, xr])

    #x = MaxPooling2D() (x)
    x = Flatten() (x)
    embedding_output = Lambda(lambda t: (K.l2_normalize(t, axis=1)) )(x) 

    embedding_model = Model(embedding_input, embedding_output)

    return embedding_model

def create_embedding_train_model(embedding):
    """ Creates a convolutional neural network (CNN) to train the imput embedding model
    Args:
        embedding: A Keras convultional neural network model
    Returns:
        embedding_train_model: A Keras convolutional neural network model
    """

    input_vector = Input(shape=(3, image_height, image_width, image_n_channels))
    
    branches = []

    for i in [0,1,2]:
        x = Lambda(lambda t: t[:,i])(input_vector)
        x = Reshape((image_height, image_width, image_n_channels)) (x)
        x = embedding (x) 
        branches.append(x)

    #x = Concatenate() ([branches[0], branches[1]])
    x1 = Subtract() ([branches[0], branches[1]])
    x1 = Lambda(lambda t: np.square(t) )(x1)
    x1 = Lambda(lambda t: K.sum(t, axis=1, keepdims=True) )(x1)

    x2 = Subtract() ([branches[0], branches[2]])
    x2 = Lambda(lambda t: np.square(t) )(x2)
    x2 = Lambda(lambda t: K.sum(t, axis=1, keepdims=True) )(x2)

    x = Subtract() ([x1, x2])
    x = Lambda(lambda t: K.sum(t, axis=1, keepdims=True) )(x)
    output = Lambda(lambda t: K.clip(t, -.1, 1) )(x)

    model = Model(input_vector, output)

    model.compile(optimizer='nadam', loss=lossMean, metrics=[lossMax])

    return model

def lossMean(yTrue, yPred):
    return K.mean(yPred)

def lossMax(yTrue, yPred):
    return K.max(yPred)

def create_relation_detection_model(embedding):
    """ Creates a convoluational neural network (CNN) using the embedding model
    and to detect if two images belong to people who are blood related
    Args:
        embedding: A Keras convultional neural network model
    Returns:
        detection_model: A Keras convultional neural network model
    """
    
    input_vector = Input(shape=(2, image_height, image_width, image_n_channels))

    branches = []
    for i in [0, 1]:
        x = Lambda(lambda t: t[:,i])(input_vector)
        x = Reshape((image_height, image_width, image_n_channels)) (x)
        x = embedding (x) 
        branches.append(x)

    #x = Concatenate() ([branches[0], branches[1]])
    x = Subtract() ([branches[0], branches[1]])
    x = Lambda(lambda t: np.square(t) )(x)
    x = Dense(64, activation='elu') (x)
    x = Dropout(0.0) (x)
    x = Dense(2) (x)
    output = Activation('softmax')(x)

    detection_model = Model(input_vector, output)
    detection_model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return detection_model


def main():

    dataset = read_dataset()
    
    X_train, X_test = prepare_data(dataset.X_train, dataset.X_test)
    y_train, y_test = dataset.y_train, dataset.y_test
    relation_dict   = dataset.relation_dict
    X_train = X_train[:num_train_samples]
    y_train = y_train[:num_train_samples]
    X_test = X_test[:num_test_samples]
    y_test = y_test[:num_test_samples]

    if os.path.isfile(RELATION_DETECTION_EMBEDDING_MODEL_PATH):
        embedding = load_model(RELATION_DETECTION_EMBEDDING_MODEL_PATH)
    else:
        embedding = emedding_model()
        embedding.summary()
        embedding_train_model = create_embedding_train_model(embedding)
        embedding_train_model.summary()
        
        train_data_generator = embedding_data_generator(X_train, y_train, num_train_samples, relation_dict, batch_size)
        test_data_generator = embedding_data_generator(X_test, y_test, num_train_samples, relation_dict, batch_size)

        embedding_train_model.fit_generator(train_data_generator, validation_data=test_data_generator, epochs=n_epochs, steps_per_epoch=num_train_samples/batch_size, validation_steps=num_test_samples/batch_size, verbose=1)
        embedding.save(RELATION_DETECTION_EMBEDDING_MODEL_PATH)
    
    if os.path.isfile(RELATION_DETECTION_CNN_MODEL_PATH):
        relation_dtection = load_model(RELATION_DETECTION_CNN_MODEL_PATH)
    else:
        relation_detction = create_relation_detection_model(embedding)
        relation_detction.summary()
        train_data_generator = data_generator(X_train, y_train, num_train_samples, relation_dict, batch_size)
        test_data_generator = data_generator(X_test, y_test, num_train_samples, relation_dict, batch_size)
    
        relation_detction.fit_generator(train_data_generator, validation_data=test_data_generator, epochs=n_epochs, steps_per_epoch=num_train_samples/batch_size, validation_steps=num_test_samples/batch_size, verbose=1)
        relation_detction.save(RELATION_DETECTION_CNN_MODEL_PATH)

    


if __name__ == "__main__":
    main()