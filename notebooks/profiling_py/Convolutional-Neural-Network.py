#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rust_machine_learning
from syed_ml_lib import *
import os


# In[2]:


# Setup the CNN
num_classifications = 10
input_height = 128
input_width = 128
input_depth = 3
CNN = rust_machine_learning.ConvolutionalNeuralNetwork(
    num_classifications, input_height, input_width, input_depth
)

# Setup the layers

# The api has 3 functions
# add_convolutional_layer(filter_height, filter_width, num_filters)
# add_max_pool_layer()
# add_fully_connected_layer()

# Add fully connected layer can only be called once, and must be called last
CNN.add_convolutional_layer(3, 3, 8)
CNN.add_max_pool_layer()
CNN.add_convolutional_layer(3, 3, 16)
CNN.add_max_pool_layer()
CNN.add_convolutional_layer(3, 3, 32)
CNN.add_max_pool_layer()
CNN.add_fully_connected_layer()

image_directory = "../../data/animals-10/"
image_directory_path = os.path.abspath(image_directory)
CNN.set_image_loader(image_directory_path, input_height, input_width)


# In[4]:


# In[5]:


animal_image_loader = rust_machine_learning.ImageBatchLoader(
    image_directory_path, input_width, input_height
)


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


# In[7]:


def train_using_set_img_loader():
    learning_rate = 1e-5
    batch_size = 4
    num_iter = 100
    print("Accuracy before training:")
    test_animals()
    print("Training...")
    CNN.train_using_image_loader(learning_rate, batch_size, num_iter)
    test_animals()


# In[8]:


# Now train the model
# train_using_raw_data()
train_using_set_img_loader()


# In[9]:


# Interactive testing (for now animal only)
images, label = animal_image_loader.batch_sample(1)

prediction = CNN.classify(images)
prediction = prediction[0]

# Assert animal image loader and CNN are in sync
assert (
    animal_image_loader.get_classifications_map()
    == CNN.get_image_loader_classification_map()
)

# Now reverse mapping from string->float to float->string
classification_map = {}
for key, value in animal_image_loader.get_classifications_map().items():
    classification_map[value] = key

print("Label: " + classification_map[label[0]] + " " + str(label[0]))
print("Prediction: " + classification_map[prediction] + " (" + str(prediction) + ")")
view_image(np.array(images[0]), input_width, input_height, colorscale="")
