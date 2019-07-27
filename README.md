# Blood related people detection using Convolutional Neaural Networks (CNN)

The aim of this project is to, given two face images, predict whthere they are blood related or not.
This project is based on the following Kaggle competeion:
https://www.kaggle.com/c/recognizing-faces-in-the-wild/overview

## Model Description

A Face Embedding model is first created which consists of three block of of convoution-max_poll-convoution-max_poll with a shortcut convolution. Fig. 1 shows the Face Embedding model. The notaion (X, Y x Y, S) denotes that the layer has X filters and uses Y x Y kernel with stride S.

The embedding model is first trained using a triplet loss (Similar to Google FaceNet) to maximize the distance between images that belong to unrelated people and mimiize the distance between images of related people. That is, given tripled training sets of (image_i, related_image_i, unrelated_image_i ) loss function is defined as:
<img src="https://latex.codecogs.com/svg.latex?\Large&space;loss=\sum_i{[||Embedding(Image_i)-Embedding(RelatedImage_i)||^2-||Embedding(Image_i)-Embedding(UnrelatedImage_i)||^2+\alpha]_+}" title="\Large" />
where $\alpha$ denotes the enforced margin between positive and negative pairs.

The Embedding model is then used to create the detection model (transfer learning) as depeicted in Fig. 2.



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
					        |----------------------| 
					        | Lambda l2_normalize  | 
					        |----------------------|
						           |
	 				        Embedded Image Vector (128)

					      Fig.1: Embedding Model



					   Input Image 1          Input Image 2
						|                      |
					|-----------------|   |-----------------|
					| Embedding Model |   | Embedding Model |
					|-----------------|   |-----------------|
						|                     |
					     |--------------------------|
					     |          Subtract        |
					     |--------------------------|
							   |
						   |-----------------|
						   | Lambda Square   |
						   |-----------------|
							   |
						    |-------------|
						    | Dense (256) |
						    | ACT: ELU    |
						    |-------------|
							   |
						   |---------------|
						   | Dropout (50%) |
						   |---------------|
							   |
						    |-------------|
						    | Dense (64)  |
						    | ACT: ELU    |
						    |-------------|
							   |
						   |---------------|
						   | Dropout (50%) |
						   |---------------|
							   |
						    |-------------|
						    | Dense (2)   |
						    |-------------|
							   |
						|----------------------|
						|        Softmax       |
						|----------------------|
						      |	         |
						  Related   Not related
					           
					  Fig.2: Blood Related Detection Model



 
## How to use the model

### train.py
Creats a model and trains it based on the Face-in-Wild image dataset (https://www.kaggle.com/c/recognizing-faces-in-the-wild/data) to detect if two input images belong to blood-related people or not.

Example usage:
```
$ python3 train.py
```
Note a trained model based on face images from Face-in-Wild dataset is included in this repository.


### predict.py
Returns the probabability that the two input images belong to blood-related people.

Example usage:
```
$ python3 predict.py -p1 "./data/test/face1.jpg" -p2 "./data/test/face2.jpg"
```


