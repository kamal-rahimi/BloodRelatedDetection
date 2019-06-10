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

n_epochs = 1
batch_size = 10

IMAGE_AUTOENCODER_MODEL_PATH = "./model/autoencoder_model"
IMAGE_ENCODER_MODEL_PATH = "./model/encoder_model"
RELATION_DETECTION_MODEL_PATH = "./model/relation_detection_model"

FIW_DATA_FILE_PATH = "./data/family_in_wild.pickle"

num_train_samples = 24000
num_test_samples = 6700

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

def create_relation_encoder_model(X_train, X_test):
    """ Creates a convoluational neural network (CNN) and trains the model to detect facial
     emotion in an input image
    Args:
        X_train: a numpy array of the train image data
        X_test: a numpy array of the test image data
    Returns:
    """
    input_image = Input(shape=(image_height, image_width, image_n_channels))
    x = Conv2D(8,(3,3), activation='elu', padding='same')(input_image)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(16,(3,3), activation='elu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(12,(3,3), activation='elu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(8,(3,3), activation='elu', padding='same')(x)
    encoded = MaxPooling2D((2,2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='elu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(12, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='tanh', padding='same')(x)
    
    autoencoder = Model(input_image, decoded)
    autoencoder.compile(optimizer='nadam', loss='mse',  metrics=['accuracy'])
    autoencoder.summary()
    autoencoder.fit(X_train, X_train, epochs=n_epochs, batch_size=batch_size, shuffle=True, verbose=2)
    encoder = Model(input_image, encoded)

    return autoencoder, encoder

def prepare_relation_detection_data(encoder, X, y, relation_dict):
    """ Prepares training and test data for emotion detection model
    Args:
        relation_dict: a Python dictionary of people related to a person
        y: a numpy array conatining labels of each image (family_id/person_id)
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

    X_clf = []
    y_clf = []
    for idx1 in range(len(X)):
        image1 = np.expand_dims(X[idx1], axis=0)
        x1 = encoder([image1])[0]
        x1 = x1/np.linalg.norm(x1)
        for idx2 in X_related_dict[idx1]:
            image2 = np.expand_dims(X[idx2], axis=0)
            x2 = encoder([ image2 ])[0]
            x2 = x2/np.linalg.norm(x2)
            X_clf.append(np.concatenate([x1.reshape(1,-1)[0], x2.reshape(1,-1)[0]], axis=0))
            #X_clf.append(np.linalg.norm(x1.reshape(1,-1)[0] - x2.reshape(1,-1)[0]))
            y_clf.append(1)

        for idx2 in X_not_related_dict[idx1]:
            image2 = np.expand_dims(X[idx2], axis=0)
            x2 = encoder([ image2 ])[0]
            x2 = x2/np.linalg.norm(x2)
            X_clf.append(np.concatenate([x1.reshape(1,-1)[0], x2.reshape(1,-1)[0]], axis=0))
            #X_clf.append(np.linalg.norm(x1.reshape(1,-1)[0] - x2.reshape(1,-1)[0]))
            y_clf.append(0)
    
    X_clf = np.array(X_clf)
    y_clf = np.array(y_clf)
    
    print(X_clf.shape, y_clf.shape)

    return X_clf, y_clf

def create_relation_detection_model(X_train, y_train, X_valid, y_valid):
    """ Creates a convoluational neural network (CNN) and trains the model to detect facial
     emotion in an input image
    Args:
        X_train: a numpy array of the train image data
        X_test: a numpy array of the test image data

    Returns:
    """
    num_features = X_train.shape[1]
    input_vector = Input(shape=(num_features,))
    x = Dense(256, activation='elu') (input_vector)
    x = Dropout(0.5) (x)
    x = Activation('elu')(x)
    x = Dense(100, activation='elu') (x)
    x = Dropout(0.0) (x)
    x = Dense(2) (x)
    output = Activation('softmax')(x)

    detect_model = Model(input_vector, output)
    detect_model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    detect_model.summary()
    detect_model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_data=(X_valid,y_valid))
    #detect_model.fit_generator(generator, validation_data=validation_data, epochs=n_epochs, steps_per_epoch=num_train_samples/batch_size, validation_steps=num_test_samples/batch_size, verbose=2)

    return detect_model

def create_relation_dictionaris(X, y, relation_dict,):
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


def data_generator(X, y, relation_dict, batch_size, data_weight, return_index=False):
    
    X_related_dict, X_not_related_dict = create_relation_dictionaris(X, y, relation_dict)

    while True:
        X_clf = []
        y_clf = []
        for idx1 in range(len(X)):
            for idx2 in X_related_dict[idx1]:
                if ( data_weight[idx1, idx2] < np.random.uniform() ):
                    continue
                X_clf.append([ X[idx1], X[idx2] ] )
                y_clf.append(1)
                if ( len(y_clf) == batch_size ):
                    if return_index:
                        yield(np.array(X_clf), np.array(y_clf), idx1, idx2)
                    else:
                        yield(np.array(X_clf), np.array(y_clf))
                    X_clf = []
                    y_clf = []

            for idx2 in X_not_related_dict[idx1]:
                if ( data_weight[idx1, idx2] < np.random.uniform() ):
                    continue
                X_clf.append([ X[idx1], X[idx2] ] )
                y_clf.append(0)
                if ( len(y_clf) == batch_size ):
                    if return_index:
                        yield(np.array(X_clf), np.array(y_clf), idx1, idx2)
                    else:
                        yield(np.array(X_clf), np.array(y_clf))
                    X_clf = []
                    y_clf = []

def create_relation_detection_model2(generator, validation_data, num_features):
    """ Creates a convoluational neural network (CNN) and trains the model to detect facial
     emotion in an input image
    Args:
        X_train: a numpy array of the train image data
        X_test: a numpy array of the test image data

    Returns:
    """
    input_vector = Input(shape=(2, image_height, image_width, image_n_channels))
    branches = []

    res1 = Conv2D(16,(4,4), strides=(4, 4), activation='elu', padding='same')
    conv1 = Conv2D(8,(2,2), activation='elu', padding='same')
    pool1 = MaxPooling2D((2,2), padding='same')
    conv2 = Conv2D(16,(2,2), activation='elu', padding='same')
    pool2 = MaxPooling2D((2,2), padding='same')
    res2 = Conv2D(64,(4,4), strides=(4, 4), activation='elu', padding='same')
    conv3 = Conv2D(32,(2,2), activation='elu', padding='same')
    pool3 = MaxPooling2D((2,2), padding='same')
    conv4 = Conv2D(64,(2,2), activation='elu', padding='same')
    pool4 = MaxPooling2D((2,2), padding='same')
    flatten = Flatten()
    #bn = BatchNormalization(axis=1)
    for i in [0,1]:
        x = Lambda(lambda t: t[:,i])(input_vector)
        print(x.shape)
        x = Reshape((image_height, image_width, image_n_channels)) (x)
        print(x.shape)
        xr = res1 (x)
        x = conv1 (x)
        x = pool1 (x)
        x = conv2  (x)
        x = pool2 (x)
        x = Add()([x, xr])
        
        xr = res2 (x)
        x = conv3 (x)
        x = pool3 (x)
        x = conv4 (x)
        x = pool4 (x)
        x = Add()([x, xr])

        x = flatten (x)
        branches.append(x)

    #x = Concatenate() ([branches[0], branches[1]])
    x = Subtract() ([branches[0], branches[1]]) 
    x = Dense(256, activation='elu') (x)
    x = Dropout(0.5) (x)
    x = Activation('elu')(x)
    x = Dense(100, activation='elu') (x)
    x = Dropout(0.5) (x)
    x = Dense(2) (x)
    output = Activation('softmax')(x)
    #x = Subtract() ([branches[0], branches[1]]) 
    #x = Lambda(lambda t: K.l2_normalize(t)) (x)
    #x = Dense(2) (x)
    #output = Activation('softmax')(x)

    detect_model = Model(input_vector, output)
    detect_model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    detect_model.summary()
    detect_model.fit_generator(generator, validation_data=validation_data, epochs=n_epochs, steps_per_epoch=num_train_samples/batch_size, validation_steps=num_test_samples/batch_size, verbose=2)    

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

    print("Train size: {}".format(len(X_train)))
    print("Test size: {}" .format(len(X_test)))

    if os.path.isfile(IMAGE_ENCODER_MODEL_PATH):
        autoencoder_model = load_model(IMAGE_AUTOENCODER_MODEL_PATH)
        encoder_model = load_model(IMAGE_ENCODER_MODEL_PATH)
    else:
        autoencoder_model, encoder_model = create_relation_encoder_model(X_train, X_train)
        autoencoder_model.save(IMAGE_AUTOENCODER_MODEL_PATH)
        encoder_model.save(IMAGE_ENCODER_MODEL_PATH)

    if os.path.isfile(RELATION_DETECTION_MODEL_PATH):
        relation_dtection = load_model(RELATION_DETECTION_MODEL_PATH)
    else:
        encoder = K.function([autoencoder_model.layers[0].input], [autoencoder_model.layers[8].output])
        X_train_clf, y_train_clf = prepare_relation_detection_data(encoder, X_train, y_train, dataset.relation_dict)
        X_test_clf, y_test_clf  = prepare_relation_detection_data(encoder, X_test, y_test, dataset.relation_dict)
        relation_detction = create_relation_detection_model(X_train_clf, y_train_clf, X_test_clf, y_test_clf)
        relation_detction.save(RELATION_DETECTION_MODEL_PATH)

    data_weights = .5*np.ones((len(X_train), len(X_train)), np.float32)
    iterate_train_data_genrator = data_generator(X_train, y_train, dataset.relation_dict, 1, data_weights, return_index=True)

    train_data_generator = data_generator(X_train, y_train, dataset.relation_dict, batch_size, data_weights)
    test_data_generator = data_generator(X_test, y_test, dataset.relation_dict, batch_size, data_weights)
    relation_detction = create_relation_detection_model2(generator=train_data_generator, validation_data=test_data_generator, num_features=256)
    
 

"""
    images = X_train[:10]
    images_rec = autoencoder_model.predict(images)
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
"""
    
    #train_relation_model(autoencoder, X_train, X_test, y_train, y_test)
    


if __name__ == "__main__":
    main()