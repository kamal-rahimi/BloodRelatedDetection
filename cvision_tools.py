"""
Methods to read and preprocess image data
"""

import numpy as np
import os
import cv2

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

image_height = 100
image_width = 100
image_n_channels = 1 

def read_image(image_path):
    """ Reads an image from an image path
    Args:
        image_path: a string indicating the image path
    Returns:
        image: a nummpy array of the image data
    """ 
    image = cv2.imread(image_path)

    return image


face_cascade = cv2.CascadeClassifier('./cv2/data/haarcascade_frontalface_default.xml')
def detect_face(image):
    """ Detecs face in an image
    Args:
        image: a numpy array containing image data
    Return:
       face[0]: A box indicating  the location of one face in the image (x, y, w, h)
    """
    faces = face_cascade.detectMultiScale(image, 1.1, 5)
    #print(faces)
    if faces != ():
        return faces[0]
    else:
        return (0, 0, image.shape[0], image.shape[1])
    #return faces[0] if faces != () else image 


def crop_face(image):
    """ Crops the face part of input image
    Args:
        image: a numpy array containing image data
    Return:
        cropped_face: the cropped face image
    """
    x, y, w, h = detect_face(image)
    cropped_face = image[y:y+h, x:x+w]
    return cropped_face


def convert_to_gray(image):
    """ Converts to grayscale
    Args:
        image: a numpy array containing image data
    Return:
        gray_image: the image in grayscale format
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def resize_with_pad(image, height, width):
    """ Resizes the image without changing scale (pads are added if necessary)
    Args:
        image: a numpy array containing image data
        height: ouput image height
        width: output image width
    Return:
        resized_image: resized image
    """
    h, w, _ = image.shape
    top_pad, bottom_pad, left_pad, right_pad = (0, 0, 0, 0)
    if ( h / w < height / width):
       pad = height / width * w - h
       top_pad = int(pad / 2)
       bottom_pad = int(pad / 2)
    else:
        pad = width / height * h - w
        right_pad = int(pad / 2)
        left_pad = int(pad / 2)

    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0 ,0, 0])

    resized_image = cv2.resize(padded_image, (height, width))

    return resized_image


def over_sample(X, y):
    random_over_sampler = RandomOverSampler(random_state=42)
    nsamples, nx, ny, nz = X.shape
    d2_X = X.reshape((nsamples, nx*ny*nz))
    d2_X_os, y_os = random_over_sampler.fit_resample(np.array(d2_X), y)
    X_os = d2_X_os.reshape(-1, nx, ny, nz)
    return X_os, y_os

def under_sample(X, y, sampling_strategy='auto'):
    random_over_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    nsamples, nx, ny, nz = X.shape
    d2_X = X.reshape((nsamples, nx*ny*nz))
    d2_X_os, y_os = random_over_sampler.fit_resample(np.array(d2_X), y)
    X_os = d2_X_os.reshape(-1, nx, ny, nz)
    return X_os, y_os

def main():
    pass


def __init__():
    pass
