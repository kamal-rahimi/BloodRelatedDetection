""" This file defines FamilyInWildDataset class to read images from
Family In Wild (FIW) image dataset
"""

import numpy as np
import pandas as pd
import os
import cv2

from cvision_tools import detect_face, crop_face, convert_to_gray, resize_with_pad

from sklearn.model_selection import train_test_split

FIW_IMAGE_PATH = "data/recognizing-faces-in-the-wild/train/"
FIW_INFO_PATH  = "data/recognizing-faces-in-the-wild/train_relationships.csv"


class FamilyInWildDataset():
    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test  = None
        self.y_train = None
        self.y_valid = None
        self.y_test  = None
        self.relation_dict = None
    
    def read(self, test_size=0.2, valid_size=0.1, gray=True, image_height=224, image_width=224, image_n_channels=1 ):
        images, labels = self.read_fiw_data()
        images = np.array(images)
        images = [resize_with_pad(image, image_height, image_width) for image in images]
        if gray==True:
            images = [convert_to_gray(image) for image in images]
            images = [image.reshape(image_height, image_width, 1) for image in images]
        images = np.array(images)
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=42)

        self.X       = images
        self.y       = labels
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test  = X_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test  = y_test

    def read_fiw_data(self):
        images = []
        labels = []
        for family in os.listdir(FIW_IMAGE_PATH):
            family_image_path = os.path.join(FIW_IMAGE_PATH, family)
            
            for person in os.listdir(family_image_path):
                person_folder_image_path = os.path.join(family_image_path, person)
                image_files = os.listdir(person_folder_image_path)
                for image_file in image_files:
                    person_image_file_path = os.path.join(person_folder_image_path, image_file)

                    person_image_file_full_path = os.path.abspath(person_image_file_path)
                    image = cv2.imread(person_image_file_full_path)
                    label = [family+'/'+person]

                    images.append(image)
                    labels.append(label)
            
        return images, labels
    
    def read_relations(self):
        relations_df = pd.read_csv(FIW_INFO_PATH)
        unique_people = list(set(relations_df["p1"]))
        relation_dict = {}
        for person in unique_people:
            related_people = relations_df[relations_df["p1"]==person]
            relation_dict[person] = list(related_people["p2"])
        self.relation_dict = relation_dict

def main():
    dataset = FamilyInWildDataset()
    dataset.read(gray=False)
    print(dataset.X_train.shape)
    for index in range(100,200):
        cv2.imshow("imag", dataset.X_train[index])
        cv2.waitKey(0)


if __name__ == "__main__":
    main()