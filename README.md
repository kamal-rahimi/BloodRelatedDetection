# Blood related people detection using Convolutional Neaural Networks (CNN)

The aim of this project is to, given two face images, predict whthere they are blood related or not.
This project is based on the following Kaggle competeion:
https://www.kaggle.com/c/recognizing-faces-in-the-wild/overview

## Model Description

The model consists of a Face Embedding step followed by a Classification step. Fig. 1 shows the Face Embedding model, which consists of three block of of convoution-max_poll-convoution-max_poll with a shortcut convolution.

The embedding model is first trained using a triplet loss (Similar to Google FaceNet) to maximize the distance between images that belong to unralted people and mimiize the distance between related people.

Fig. 2 shows how the Embedding model is trained.

The detection model is depeicted in Fig. 3, where Embedding model is resuled through transfer leraing. 
 


## How to use the model

### train_gender.py



### predict.py


