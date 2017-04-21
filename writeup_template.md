#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.png "Histogram"
[image2]: ./examples/normalized.png "Normalized Image"
[image3]: ./examples/augmentation.png "Augmented Image"
[image4]: ./examples/go_straight_or_right.png "Traffic Sign 1"
[image5]: ./examples/no_entry.png "Traffic Sign 2"
[image6]: ./examples/right_of_way_at_next_intersection.png "Traffic Sign 3"
[image7]: ./examples/speed_limit_30.png "Traffic Sign 4"
[image8]: ./examples/stop.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/agarwa65/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The code for the step is in the third and fourth cell of the notebook. 

It is a list of the unique classes present in the dataset. Though each labeled image might have different frequency in the training, validation and test data set.  Some signs are more frequent than the other representing real work characteristics. 

The image below shows the histogram distribution of the labels over datasets:
![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Pre-processing technique - I tried the Normalizing image data using image/255.0 - 0.5. But later on used another normalization technique to get the dataset with zero mean and unit variance. This seemed to have worked better.
Normalizing the image causes the pixel values to be in the range 0-1 from 0-255, which ensures that the network is trained on the same scale. This makes the model more robust.

Here is an example of a traffic sign image before and after normalizing.

![alt text][image2]

I decided to generate additional data to make the model more robust by  increasing the samples for training. The augmented dataset is generated using keras library ImageDataGenerator, which randomly introduces rotation, x and y translation, zoom and sheer factor to the original data. This makes the training data set have 69598 images. The validation and test image dataset are the same as original.

Here is an example of an original image and an augmented image:

![alt text][image3]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the modified LeNet Architecture:
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Dropout	        	| Keep probablility =0.52 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Dropout	        	| Keep probablility =0.52 				|
| Flatten       		| Input = 5x5x16. Output = 400. 					|
| Fully connected		| Input = 400. Output = 120. 					|
| Dropout	        	| Keep probablility =0.52 				|
| Fully connected		| Input = 120. Output = 120. 					|
| Dropout	        	| Keep probablility =0.52 				|
| Fully connected		| Input = 120. Output = 84. 					|
| Dropout	        	| Keep probablility =0.52 				|
| Fully connected		| Input = 84. Output = 43. 					|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The cells 9- 13 contain the code for model architecture, training pipeline and the model evaluation.

To train the model, I used:
Epochs : 40
Batch size : 64
Softmax cross entropy function 
Adam Optimizer for learning rate with learning rate = 0.0009
The above parameters worked best to get a good training accuracy. I added another fully connected layer to the architecture for it to be able to process all the training data.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 95.8%
* validation set accuracy of 94.1% 
* test set accuracy of 93.3%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Steps:
1. Preprocessed the data by converting ot grayscale and then normalizing the data. Grayscale data was not improving the model accuracy, hence I just normalized the images.
2. Augmented the training data by using keras library as mentioned above.
3. Used the modified LeNet architecture for this project with an added fully connected layer, with changes to handle color images. 
4. Used dropout as the model was overfitting, tried different probablities based on the overfitting but k_prob = 0.52 worked the best.
5. Different learning rates, batch sizes and epochs were also iterated upon.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, they havebeen modified with random rotation, translation or brightness change, to make it more realistic for the model:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of way at the next intersection		| Right of way at the next intersection		| 
| NO Entry    			| Turn left ahead 										|
| Go straight or right			| NO Vehicles									|
| 30 km/h	      		| 30 km/h					 				|
| Stop			| Turn Right ahead  							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The trained was then tested on the new five german traffic signs and the predictions were printed. Out of five, two were correctly predicted. The model could be further improved for better performance.

For the first image, the model is relatively sure that this is a Right of way at the next intersection (probability of 0.566), and the image does contain a Right of way at the next intersection. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Right of way at the next intersection   		| 
| .17     				| Pedestrains 										|
| .13					| Beware of ice/snow									|
| .04	      			| ahead only					 				|
| .03				    | Road Work     							|


For the second image NO entry, it was wrong and did not guess no entry at all. But all the signs were mostly similar looking.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Turn left ahead   		| 
| .17     				| Ahead only 								|
| .13					| Keep right								|
| .04	      			| Go straight or right		 				|
| .03				    |  Go straight or left     					|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



