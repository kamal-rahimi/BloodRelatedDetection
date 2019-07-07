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


RELATION_DETECTION_CNN_MODEL_PATH = "./model/blood_related_detect_model"
RELATION_DETECTION_EMBEDDING_MODEL_PATH = "./model/blood_related_detect_embedding"

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
        dataset.read(image_height=image_height, image_width=image_width)
        dataset.read_relations()
        pickle.dump(dataset, open(FIW_DATA_FILE_PATH, 'wb'))
    return dataset


def prepare_data(X_train, X_test):
    """ Prepares training and test data for family relations detection model from Family in Wild (FIW)
    image dataset 
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
    """ Creates a dictionary containing the ID of images that are blood related to an image ID and
    a dictionary (with same size) containing the ID of images that are not related the image
    Args:
        X: numpy array of data
        y: numpy array of labels
        relation_dict: a dictionary containing people who are blood related
    Returns:
        X_related_dict: a dictionary containing the ID of images that are blood related to the image
        X_not_related_dict: a dictionary containing the ID of images that are not related  to the image
    """
    num_indices = len(y)

    X_related_dict = {}
    X_not_related_dict = {}
    num_related = 0
    num_not_related = 0
    
    for idx1 in range(num_indices):
        X_related_dict[idx1] = []
        X_not_related_dict[idx1] = []
        for idx2 in range(num_indices):
            if ( (y[idx2] == y[idx1]) and (idx1 != idx2)) or ((y[idx1][0] in relation_dict) and (y[idx2][0] in relation_dict[y[idx1][0]]) ):
                X_related_dict[idx1].append(idx2)
                num_related += 1
        for idx2 in range(max(0, idx1-30), num_indices):
            if ( len(X_not_related_dict[idx1]) == len(X_related_dict[idx1]) ):
                    break;
            elif (y[idx2] != y[idx1]) and ( (y[idx1][0] not in relation_dict) or (y[idx2][0] not in relation_dict[y[idx1][0]]) ):
                X_not_related_dict[idx1].append(idx2)
                num_not_related += 1
                
    print("Data size:", num_related, num_not_related)
    
    return X_related_dict, X_not_related_dict

def data_generator(X, y, data_len, relation_dict, batch_size):
    """ Creates a datagenrator to feed two images and label (related or not related) to the Keras model
    Args:
        X: input images
        y: input image lables
        relation_dict: a python dictionary containing relation information
        data_weight: weights used to sample imput data
        return_index: weather return indices of the two image along with data or not
    Returns:
         yields with numpy arrays contining two pair of images and lable (related, not related)
    """
    X_related_dict, X_not_related_dict = create_relation_dictionaris(X, y, relation_dict)
    X_clf = []
    y_clf = []
    while True:
        #print("\n rel:", rel,"nrel:", nrel, "\n")        
        for idx1 in range(len(X)):
            for idx2 in X_related_dict[idx1]:
             #   weight = num_train_samples * data_weight[idx1, idx2] /2
              #  if ( weight < np.random.uniform() ):
              #      continue
                X_clf.append([ X[idx1], X[idx2] ] )
                y_clf.append([1])
                if ( len(y_clf) == batch_size ):
                    yield(np.array(X_clf), np.array(y_clf))
                    X_clf = []
                    y_clf = []

            random_index = list(np.random.randint(len(X), size=20))
            idx3_in = 0

            for i in range(len(X_related_dict[idx1])):
                while random_index[idx3_in] in X_related_dict[idx1]:
                    idx3_in += 1
                idx3 = random_index[idx3_in]

                X_clf.append([ X[idx1], X[idx3] ] )
                y_clf.append([0])
                if ( len(y_clf) == batch_size ):
                    yield(np.array(X_clf), np.array(y_clf))
                    X_clf = []
                    y_clf = []

def embedding_data_generator(X, y, data_len, relation_dict, batch_size):
    """ Creates a datagenrator to feed two images and label (related or not related) to the Keras model
    Args:
        X: input images
        y: input image lables
        relation_dict: a python dictionary containing relation information
        data_weight: weights used to sample imput data
        return_index: weather return indices of the two image along with data or not
    Returns:
         yields with numpy arrays contining two pair of images and lable (related, not related)
    """
    X_related_dict, X_not_related_dict = create_relation_dictionaris(X, y, relation_dict)
    X_clf = []
    y_clf = []
    num_data = 0
    while True:
        #print("\n rel:", rel,"nrel:", nrel, "\n")
        #print("\n num_data:", num_data, "\n")      
        for idx1 in range(len(X)):
            random_index = list(np.random.randint(len(X), size=20))
            #random.shuffle(random_index)
            #random.shuffle(X_related_dict[idx1])
            idx3_in = 0
            for idx2 in X_related_dict[idx1]:
                while random_index[idx3_in] in X_related_dict[idx1]:
                    idx3_in += 1
                idx3 = random_index[idx3_in]
            #for idx2, idx3 in zip(X_related_dict[idx1], X_not_related_dict[idx1]):
            #for idx2 in X_related_dict[idx1]:
            #    for idx3 in X_not_related_dict[idx1]:
                X_clf.append([ X[idx1], X[idx2], X[idx3] ] )
                y_clf.append([-1])
                num_data += 1
                if ( len(y_clf) == batch_size ):
                    yield(np.array(X_clf), np.array(y_clf))
                    X_clf = []
                    y_clf = []

def emedding_model():
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
    #x = BatchNormalization(axis=1) (x)
    embedding_output = Lambda(lambda t: (K.l2_normalize(t, axis=1)) )(x) 

    embedding_model = Model(embedding_input, embedding_output)

    return embedding_model

def create_embedding_model():
    """ Creates a convoluational neural network (CNN) and trains the model to detect facial
     emotion in an input image
    Args:
        num_features: Number of input data features
    Returns:
        cnn_model: A Keras convultional neural network
    """
    embedding = emedding_model()

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
    output = Lambda(lambda t: K.clip(t, -.2, 10) )(x)

    model = Model(input_vector, output)

    model.compile(optimizer='nadam', loss=embeddingLoss, metrics=[embeddingLoss2])

    return model, embedding

def embeddingLoss(yTrue, yPred):
    #return (K.max(yPred)/ (K.max(yPred) - K.mean(yPred) + .000000001 ))
    return K.mean(yPred)

def embeddingLoss2(yTrue, yPred):
    return K.max(yPred)

def create_relation_detection_model(embedding):
    """ Creates a convoluational neural network (CNN) and trains the model to detect facial
     emotion in an input image
    Args:
        num_features: Number of input data features
    Returns:
        cnn_model: A Keras convultional neural network
    """
    
    input_vector = Input(shape=(2, image_height, image_width, image_n_channels))
    
    for l in embedding.layers: 
        l.trainable=False

    branches = []
    for i in [0,1]:
        x = Lambda(lambda t: t[:,i])(input_vector)
        x = Reshape((image_height, image_width, image_n_channels)) (x)
        x = embedding (x) 
        branches.append(x)

    x = Concatenate() ([branches[0], branches[1]])
    #x = Subtract() ([branches[0], branches[1]])
    #x = Lambda(lambda t: np.square(t) )(x)
    #x = Lambda(lambda t: (tf.norm(t, axis=1, keepdims=True)) )(x)
    x = Dense(256, activation='elu') (x)
    x = Dropout(0.3) (x)
    x = Dense(64, activation='elu') (x)
    x = Dropout(0.3) (x)
    x = Dense(2) (x)
    output = Activation('softmax')(x)

    cnn_model = Model(input_vector, output)
    cnn_model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return cnn_model


def main():

    dataset = read_dataset()
    
    X_train, X_test = prepare_data(dataset.X_train, dataset.X_test)
    y_train, y_test = dataset.y_train, dataset.y_test
    relation_dict   = dataset.relation_dict
    X_train = X_train[:num_train_samples]
    y_train = y_train[:num_train_samples]
    X_test = X_test[:num_test_samples]
    y_test = y_test[:num_test_samples]
    num_features = X_train.shape[1]*2

    if os.path.isfile(RELATION_DETECTION_EMBEDDING_MODEL_PATH):
        embedding = load_model(RELATION_DETECTION_EMBEDDING_MODEL_PATH)
    else:
        model, embedding = create_embedding_model()
        model.summary()
        embedding.summary()
        train_data_generator = embedding_data_generator(X_train, y_train, num_train_samples, relation_dict, batch_size)
        data_weights = 1/num_test_samples * np.ones((len(X_train), len(X_train)), np.float32)
        test_data_generator = embedding_data_generator(X_test, y_test, num_train_samples, relation_dict, batch_size)

        model.fit_generator(train_data_generator, validation_data=test_data_generator, epochs=n_epochs, steps_per_epoch=num_train_samples/batch_size, validation_steps=num_test_samples/batch_size, verbose=1)
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