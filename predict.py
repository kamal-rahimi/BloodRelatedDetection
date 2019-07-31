"""
"""

import tensorflow as tf
from tensorflow.python import keras
from keras.models import load_model

import numpy as np
import pickle
import argparse
import os

from cvision_tools import read_image, detect_face, crop_face, convert_to_gray, resize_with_pad

import cv2 as cv2

image_height = 64
image_width = 64
image_n_channels = 1

RELATION_DETECTION_CNN_MODEL_PATH = "./model/blood_related_detect_model"
RELATION_DETECTION_EMBEDDING_MODEL_PATH = "./model/blood_related_detect_embedding"

TEST_DATA_PATH = "./data/test/"


def prepare_image(image):
    image = np.array(image)
    face  = crop_face(image)
    face  = resize_with_pad(face, image_height, image_width)
    face  = convert_to_gray(face)
    face  = np.array(face)
    face  = face.reshape(image_height, image_width, image_n_channels)
    face  = face.astype('float32')
    face  = face / 128 - 1
    return image, face


def indetify_blood_relation(face1, face2):
    relation_detection = load_model(RELATION_DETECTION_CNN_MODEL_PATH)
    embedding = load_model(RELATION_DETECTION_EMBEDDING_MODEL_PATH)
    print(face1.shape)
    X = [[face1, face2]]
    X = np.array(X)
    print(embedding.predict(face1.reshape(1, image_height, image_width, 1)))
    print(embedding.predict(face2.reshape(1, image_height, image_width, 1)))
    probs = relation_detection.predict(X)[0]
    return probs

def predict_images(image1_path, image2_path, image_out_path=''):
    image1 = read_image(image1_path)
    image2 = read_image(image2_path)
    image1, face1 = prepare_image(image1)
    image2, face2 = prepare_image(image2)
  
    probs = indetify_blood_relation(face1, face2)
    print(probs)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-p1", "--path1", type=str, default="", help="Specify the path to the fisrt input image")
    ap.add_argument("-p2", "--path2", type=str, default="", help="Specify the path to the second input image")
    args = vars(ap.parse_args())
    image1_path = args["path1"]
    image2_path = args["path2"]
    
    if (image1_path != "" and image2_path != ""):
        predict_images(image1_path, image2_path)


if __name__ == "__main__":
    main()