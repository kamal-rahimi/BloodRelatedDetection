"""
Creates a convolutional neural network (CNN) in Keras and trains the network
to identify if two pictures belong to people who are blood related.
"""

import tensorflow as tf
from tensorflow.python import keras
from keras import Model
from keras import backend as K
from keras.models import load_model, save_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Dropout, Reshape, Flatten, Concatenate, Add, Subtract, Lambda


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

n_epochs = 10
batch_size = 10

num_predictors = 10
learning_rate = .4

RELATION_DETECTION_ADABOOST_MODEL_PATH = "./model/relation_detect_adaboost_model"

FIW_DATA_FILE_PATH = "./data/family_in_wild.pickle"

num_train_samples = 100000
num_test_samples = 14000



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

def create_relation_dictionaris(X, y, relation_dict,):
    num_indices = len(y)

    X_related_dict = {}
    X_not_related_dict = {}
    num_related = 0
    num_not_related = 0
    
    for idx1 in range(num_indices):
        X_related_dict[idx1] = []
        X_not_related_dict[idx1] = []
        for idx2 in range(num_indices):
            if (y[idx2] == y[idx1]) or ((y[idx1][0] in relation_dict) and (y[idx2][0] in relation_dict[y[idx1][0]]) ):
                X_related_dict[idx1].append(idx2)
                num_related += 1
        for idx2 in range(max(0, idx1-10), num_indices):
            if (y[idx2] != y[idx1]) and ( (y[idx1][0] not in relation_dict) or (y[idx2][0] not in relation_dict[y[idx1][0]]) ):
                X_not_related_dict[idx1].append(idx2)
                num_not_related += 1
                if ( len(X_not_related_dict[idx1]) == len(X_related_dict[idx1]) ):
                    break;
    print("Data size:", num_related, num_not_related)
    
    return X_related_dict, X_not_related_dict

def data_generator(X, y, relation_dict, batch_size):
    
    X_related_dict, X_not_related_dict = create_relation_dictionaris(X, y, relation_dict)
    num_yeild = 0
    while True:
        X_clf = []
        y_clf = []
        print("num_batches", num_yeild)
        for idx1 in range(len(X)):
            for idx2 in X_related_dict[idx1]:
                if ( np.random.uniform() < 0.5 ):
                    continue
                X_clf.append([ X[idx1], X[idx2] ] )
                y_clf.append(1)
                if ( len(y_clf) == batch_size ):
                    yield(np.array(X_clf), np.array(y_clf))
                    num_yeild +=1
                    X_clf = []
                    y_clf = []

            for idx2 in X_not_related_dict[idx1]:
                if ( np.random.uniform() < 0.5 ):
                    continue
                X_clf.append([ X[idx1], X[idx2] ] )
                y_clf.append(0)
                if ( len(y_clf) == batch_size ):
                    num_yeild +=1
                    yield(np.array(X_clf), np.array(y_clf))
                    X_clf = []
                    y_clf = []
           # print("num_batches", num_yeild, idx1)

def create_relation_detection_model(num_features):
    """ Creates a convoluational neural network (CNN) and trains the model to detect facial
     emotion in an input image
    Args:
        X_train: a numpy array of the train image data
        X_test: a numpy array of the test image data

    Returns:
    """
    input_vector = Input(shape=(2, image_height, image_width, image_n_channels))
    branches = []

    res1 = Conv2D(16,(8,8), strides=(8, 8), activation='elu', padding='same')
    conv1 = Conv2D(8,(4,4), activation='elu', padding='same')
    pool1 = MaxPooling2D((4,4), padding='same')
    conv2 = Conv2D(16,(2,2), activation='elu', padding='same')
    pool2 = MaxPooling2D((2,2), padding='same')
    res2 = Conv2D(32,(8,8), strides=(8, 8), activation='elu', padding='same')
    conv3 = Conv2D(24,(4,4), activation='elu', padding='same')
    pool3 = MaxPooling2D((4,4), padding='same')
    conv4 = Conv2D(32,(2,2), activation='elu', padding='same')
    pool4 = MaxPooling2D((2,2), padding='same')
    flatten = Flatten()
    #bn = BatchNormalization(axis=1)
    for i in [0,1]:
        x = Lambda(lambda t: t[:,i])(input_vector)
        x = Reshape((image_height, image_width, image_n_channels)) (x)
        xr = res1 (x)
        x = conv1 (x)
        x = pool1 (x)
        x = conv2 (x)
        x = pool2 (x)
        x = Add()([x, xr])
        
        xr = res2 (x)
        x = conv3 (x)
        x = pool3 (x)
        x = conv4 (x)
        x = pool4 (x)
        x = Add()([x, xr])

        x = flatten (x)
        x = Lambda(lambda t: (K.l2_normalize(t, axis=1)) )(x)
        branches.append(x)

    #x = Concatenate() ([branches[0], branches[1]])
    x = Subtract() ([branches[0], branches[1]])
    x = Lambda(lambda t: np.absolute(t) )(x)
    #x = Lambda(lambda t: (tf.norm(t, axis=1, keepdims=True)) )(x)
    x = Dense(256, activation='elu') (x)
    x = Dropout(0.0) (x)
    x = Activation('elu')(x)
    x = Dense(100, activation='elu') (x)
    x = Dropout(0.0) (x)
    x = Dense(2) (x)
    output = Activation('softmax')(x)

    detect_model = Model(input_vector, output)
    detect_model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return detect_model
    
def main():

    if os.path.isfile(FIW_DATA_FILE_PATH):
        dataset = pickle.load(open(FIW_DATA_FILE_PATH, 'rb'))
    else:
        dataset = FamilyInWildDataset()
        dataset.read(image_height=image_height, image_width=image_width)
        dataset.read_relations()
        pickle.dump(dataset, open(FIW_DATA_FILE_PATH, 'wb'))
    
    X_train, X_test = prepare_data(dataset.X_train, dataset.X_test)
    y_train, y_test = dataset.y_train, dataset.y_test
    relation_dict   = dataset.relation_dict

    if os.path.isfile(RELATION_DETECTION_ADABOOST_MODEL_PATH):
        relation_dtection = load_model(RELATION_DETECTION_ADABOOST_MODEL_PATH)
    else:
        num_features = X_train.shape[1]*2
        relation_detction = create_relation_detection_model(num_features)
        relation_detction.summary()
        train_data_generator = data_generator(X_train, y_train, relation_dict, batch_size)
        test_data_generator = data_generator(X_test, y_test, relation_dict, batch_size)
    
        relation_detction.fit_generator(train_data_generator, validation_data=test_data_generator, epochs=n_epochs, steps_per_epoch=num_train_samples/batch_size, validation_steps=num_test_samples/batch_size, verbose=2)
        relation_detction.save(RELATION_DETECTION_ADABOOST_MODEL_PATH)

    


if __name__ == "__main__":
    main()