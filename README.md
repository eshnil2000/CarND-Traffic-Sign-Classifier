---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/Traffic%20Classifier%20Solution.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the "length" command to calculate number of samples

.shape to find shape of image & 

unique to find number of class Labels of the traffic signs data set:

image_shape = X_train[0].shape
unique, counts = np.unique(y_train, return_counts=True)

*Number of training examples = 34799

*Number of validation examples = 4410

*Number of testing examples = 12630

*Image data shape = (32, 32, 3)

*Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data class labels are distrubuted.

![Class Label distributions](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/solution-images/histogram.png)
The histogram shows unevenly distributed samples of each class. the max number samples for any class is ~2000 while the minimum is <200, this provides for an unbalanced data set.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to reduce the training complexity/ time, also previous papers have shown grayscale images effective for training Traffic Signs.

Here is an example of a traffic sign image before and after grayscaling.

BEFORE:
![Color Image](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/solution-images/rgb.png)

AFTER:
![Grayscale Image](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/solution-images/gray.png)

As a last step, I normalized the image data by subtracting 128 from each pixel and dividing by 128, to roughly position the image data within a range of [-1,1] to reduce variations, make it easier to train. Normalized data has shown to perform better than data that has high variations in some variables versus others.

I decided to generate additional data because after all these steps, the accuracy on the validation data set was <93%.

To add more data to the the data set, I used the following techniques: 
I calculated the maximum number of samples of the 43 labels. I then eqaulized the number of samples for each and every Class label to that maximum value (in my case, ~2000 samples each) to make for a uniformly distributed sample size of each label.

The technique I used to augment the data set of each class label was very simple:
I created an array of indices for each class Label (for example, class Label "0" had 180 samples, so I gathered those 180 indices). I then took the difference between the maximum number of samples in ny class label (in this case, 2000) and found the difference (1820). I then uniformly randomly sampled those 180 images 1800 times and appended 1800 more samples of that class, and appended the corresponding Label. I repeated this for each an every class to equalize the number of samples.

This led to a 148% increase in the data set from about ~35k images to about ~85k images.

Here is an example of an CLass label distribution histogram after augmenting with this additional data:

![Grayscale Image](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/solution-images/histogram_augmented.png)

Surprisingly, even without advanced data augmentation techniques, this simple technique was able to boost the accuracy above the 93% mark. To increase the accuracy of the validation set even further, one could try out more sophisticated Data Augmentation techniques like rotating images, blurring images, linear translating images,etc to add to the number of samples. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet model as described in Yan LeCun's implementation. Some changes were made to accomodate image size, adjust for color channels, adjust for the number of final output classes. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, 1 input channel, 6 output channels, VALID padding, output 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, output 14x14x6  				|
| Convolution 5x5	    | 1x1 stride,  6 input channel, 16 output channels, VALID padding, output 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, output 5x5x16 				|
| Flatten		| 5x5x16=400      									|
| Fully connected		| input=400, Output=120        									|
| RELU					|												|
| Fully connected		| input=120, Output=84        									|
| RELU					|												|
| Fully connected		| input=120, Output=84        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

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
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


