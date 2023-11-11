#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rust_machine_learning
from syed_ml_lib import *
import os


# In[2]:


# Setup the CNN
num_classifications = 10
input_width = 28
input_height = 28
input_depth = 1
filters_per_conv_layer = [10]
filter_dimension = 3

CNN = rust_machine_learning.SimplifiedConvolutionalNeuralNetwork(num_classifications, input_width, input_height, input_depth, filters_per_conv_layer, filter_dimension)

image_directory = "../data/animals-10/"
image_directory_path = os.path.abspath(image_directory)
CNN.set_image_loader(image_directory_path, input_width, input_height)


# In[3]:


# Grab mnist data
import random
import numpy as np

df = pd.read_csv("../data/digit-recognizer/train.csv")

# Normalize images
df.iloc[:, 1:] = df.iloc[:, 1:].div(255.0).sub(0.5)
images = df.iloc[:, 1:].values.tolist()
df.iloc[:, 0] = df.iloc[:, 0].astype(np.float32)
labels = df.iloc[:, 0].values.tolist()

# Split into training and testing
test_images = images[0:20000]
test_labels = labels[0:20000]
train_images = images[20000:]
train_labels = labels[20000:]

df


# In[4]:


def test_mnist():
    num_batches_to_test = 500
    num_right = 0
    for i in range(num_batches_to_test):

        # Random sample one image from python list of images
        rand_index = random.randint(0, len(test_images) - 1)
        image = test_images[rand_index]
        label = test_labels[rand_index]

        prediction = CNN.classify([[image]])[0]
        if prediction == label:
            num_right += 1

    print("Accuracy: " + str(100.0 * num_right / (num_batches_to_test)))


# In[5]:


animal_image_loader = rust_machine_learning.ImageBatchLoader(image_directory_path, 128, 128)
def test_animals():
    num_batches_to_test = 100
    num_right = 0
    for i in range(num_batches_to_test):

        images, label = animal_image_loader.batch_sample(1)

        prediction = CNN.classify(images)[0]
        if prediction == label[0]:
            num_right += 1

    print("Accuracy: " + str(100.0 * num_right / (num_batches_to_test)))


# In[6]:


def train_using_set_img_loader():
    learning_rate = 1e-3
    batch_size = 1
    num_iter = 100
    test_animals()
    print("Accuracy before training:")
    print("Training...")
    CNN.train_using_image_loader(learning_rate, batch_size, num_iter)
    test_animals()


# In[9]:


def train_using_raw_data():
    learning_rate = 1e-5
    batch_size = 4
    num_iter = 3000

    print("Accuracy before training:")
    test_mnist()
    print("Training...")
    for i in range(0, num_iter):

        # Create image and label batch from mnist
        image_batch, label_batch = [], []
        for j in range(batch_size):
            rand_index = random.randint(0, len(train_images) - 1)
            image = train_images[rand_index]
            label = train_labels[rand_index]
            image_batch.append([image])
            label_batch.append(label)

        # Train using MNIST dataset, select one image at a time randomly
        CNN.train_raw_data(image_batch, label_batch, learning_rate)

    print("Accuracy after training:")
    test_mnist()


# In[10]:


# Now train the model
train_using_raw_data()

