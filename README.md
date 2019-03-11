# Car_Model_Categorization
DataSet URL:
https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder

Objective:
  Through the analysis of pictures, Implement a convolutional neural network with different learning frame for visual recognition tasks classify the level of make, model, and year of a car.
  
Dataset Description:
  Stanford Car Dataset
The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S.

Works:
Using Keras and ImageDataGenerator for classification algorithms, Densnet 121 is trained for the dataset.
Keras ImageDataGenerator is used to load, transform the input image of the dataset and augment the data.
The ImageNet pretrained model is set to be the starting weights of the model. Then, final layers are added for the model: these layers use the ReLU activation function and the output layer use softmax.

Implementation:
Plain CNN Neural Network:
batch size=64
Optimization method= Stochastic Gradient Descent with Momentum=0.9
Learning rate =0.01
Epoch=100
VALID ACCURACY=0.58%

Plain CNN with Data Augmentation:
VALID ACCURACY=4.4%

VGG with ImageNet model:
VALID ACCURACY=36.5%

ResNet with ImageNet model:
VALID ACCURACY=56.42%




