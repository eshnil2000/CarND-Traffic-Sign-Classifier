---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/Traffic%20Classifier%20Solution.ipynb)

An HTML version of the Python notebook can be found here:
[HTML file](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/Traffic%20Classifier%20Solution.html)

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
I went with the Adam Optimizer for this exercise.
To train the model, I played with a few different hyperparameters: 

Dropout
Learning Rate
Batch size
No of Epochs

I tried dropout factors of 0.65/0.5 and trial & error but did not see consistent improvement, ,even though there is literature which suggests drop out (especially in fully connected layers) helps improve accuracy. I decided to leave this parameter out in the end.

I adjusted the learning rate, started with a very course number 0.1, then worked way down to 0.00001 and then finally settled on a value of 0.00097 which seemed to give sufficient performance.

I played with batch size, starting with size of 200 down to 100, settled on batch size of 150. 

I played with number of Epochs, started with large numbers, observed roughly where the accuracy flattened out then built in early stop point once I hit the desired 93% accurancy mark (around 22 Epochs)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 93.3%
* test set accuracy of 91.2%

#### PROJECT RESUBMISSION: Validation accuracy increased to 94.3%, Test set accuracy increased to 93.4%!
* Key changes: I increased the augmented data set by about 30% uniformly across all classes, increased the Epochs to 60 and changed the batch size to 200.
* I experimented with modifying the images with random small changes (translation, rotation, brightness, etc) , and I also played with the LeNet Architecture, adding in Convolution layers. I did not notice dramatic increase in accuracy, and with the increased data sizes, training on my CPU was taking >30mins. Time to switch to GPU for the next project.


I went with the LeNet architecture, because right from the outset, it seemed to give a good accuracy of >70%.
Since the test set & unseen images also gave pretty good accuracy, I could conclude reasonably well that the model was not overfitted to the data set (even with the augmented data).

The training seemed to complete in a reasonable amount of time on my Mac laptop with limited resources, so it is not overly complex, nor is the data size overwhelmingly large. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Grayscale Image](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/web-images/4.png)
![Grayscale Image](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/web-images/1.png)
![Grayscale Image](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/web-images/3.png)
![Grayscale Image](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/web-images/6.png)
![Grayscale Image](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/web-images/5.png)


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![Grayscale Image](https://github.com/eshnil2000/CarND-Traffic-Sign-Classifier/blob/master/web-images/classify.png)


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Vehicles     		| Speed Limit 50km/h   									| 
| Ahead Only     			| Ahead Only 										|
| Speed Limit 30km/h			| Speed Limit 30km/h										|
| General caution      		| General caution					 				|
| Go straight or left			| Go straight or left      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 91%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
For each of the images, the model was surprisingly 100% confident in it's prediction (including the first image, which ws incorrectly predicted)

TopKV2(values=array([[ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.]], dtype=float32), indices=array([[ 2,  0,  1,  3,  4],
       [35,  0,  1,  2,  3],
       [ 1,  0,  2,  3,  4],
       [18,  0,  1,  2,  3],
       [37,  0,  1,  2,  3]], dtype=int32))

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


