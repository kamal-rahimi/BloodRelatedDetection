# Blood related people detection using Convolutional Neaural Networks (CNN)

The aim of this project is to, given two face images, predict whthere they are blood related or not.
This project is based on the following Kaggle competeion:
https://www.kaggle.com/c/recognizing-faces-in-the-wild/overview

## Model Description

A Face Embedding model is first created which consists of three block of of convoution-max_poll-convoution-max_poll with a shortcut convolution. Fig. 1 shows the Face Embedding model.

The embedding model is first trained using a triplet loss (Similar to Google FaceNet) to maximize the distance between images that belong to unrelated people and mimiize the distance between images of related people. Fig. 2 shows how the Embedding model is trained.

The Embedding model is used to created the detection model (transfer learning) as depeicted in Fig. 3.



				     		 Input Face image (64x64x1)
					      		   |
	                     |-----------------------------|
			     |				   |
			     |			 |-------------------|
			     |			 | Conv2D (8,2x2,1)  | 
			     |			 | ACT: ELU          |
	        	     |			 |-------------------|
	        	     |	       			   |
	          	     |			|---------------------|
	           	     |			| MaxPooling2D (2x2,2)|
	           |-------------------|        |---------------------|
	           | Conv2D (16,4x4,4) | 	       	   |	
	           | ACT: ELU          | 	 |-------------------|
	           |-------------------|	 | Conv2D (16,2x2,1) | 
			     |			 | ACT: ELU          |
			     |			 |-------------------|
			     |				   |
			     |			 |-------------------|
			     |			 | MaxPooling2D (2,2)|
			     |			 |-------------------|
			     |				   |
			     |			   |----------------|
			     |---------------------|       +        |
						   |----------------|
							   |
			     |-----------------------------|
			     |				   |
			     |			 |-------------------|
			     |			 | Conv2D (32,2x2,1) | 
			     |			 | ACT: ELU          |
	        	     |			 |-------------------|
	        	     |	       			   |
	          	     |			|---------------------|
	           	     |			| MaxPooling2D (2x2,2)|
	           |-------------------|        |---------------------|
	           | Conv2D (64,4x4,4) | 	       	   |	
	           | ACT: ELU          | 	 |-------------------|
	           |-------------------|	 | conv2D (64,2x2,1) | 
			     |			 | ACT: ELU          |
			     |			 |-------------------|
			     |				   |
			     |			|---------------------|
			     |			| MaxPooling2D (2x2,2)|
			     |			|---------------------|
			     |				   |
			     |			   |----------------|
			     |---------------------|       +        |
						   |----------------|
							   |
			     |-----------------------------|
			     |				   |
			     |			  |------------------|
			     |			  | Conv2D (96,2x2,1)| 
			     |			  | ACT: ELU         |
	        	     |			  |------------------|
	        	     |	       			   |
	          	     |			 |---------------------|
	           	     |			 | MaxPooling2D (2x2,2)|
	           |--------------------|        |---------------------|
	           | Conv2D (128,4x4,4) | 	     	   |	
	           | ACT: ELU           | 	  |-------------------|
	           |--------------------|	  | Conv2D (128,2x2,1)| 
			     |			  | ACT: ELU          |
			     |			  |-------------------|
			     |				   |
			     |			 |---------------------|
			     |			 | MaxPooling2D (2x2,2)|
			     |			 |---------------------|
			     |				   |
			     |			   |----------------|
			     |---------------------|       +        |
						   |----------------|
							   |
						    |-------------| 
						    |   Flatten   | 
						    |-------------|
							  |
	 					 Embedded Vector (128)

					      Fig.1: Embedding Model



 

. 

## How to use the model

### train_gender.py



### predict.py


