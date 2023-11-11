#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rust_machine_learning
import pandas as pd
from syed_ml_lib import *


# In[2]:


# Get the csv data
# df = pd.read_csv("../data/iris.data")
# df = pd.read_csv("../data/kmeans-dataset.csv")
# df = pd.read_csv("../data/winequality-red.csv")
df = pd.read_csv("../data/digit-recognizer/train.csv")
df = df.sample(n=10000)
df  


# In[3]:


to_encode = [] # We won't encode the label column. The output layer will use 0-n (assuming classifications are ordered starting at 0), to determine error.
encoded, feature_to_encoded_cols = one_hot_encode(df, to_encode)
encoded


# In[4]:


# Remove unnecessary data or NAN
to_remove = []
encoded = encoded.drop(encoded.columns[to_remove], axis=1)
encoded


# In[5]:


# Prepare data for training with input rust functions expect
label_col = 0

# Normalize columns
labels = encoded.iloc[:, label_col]
max_per_col = encoded.abs().max().replace(0,1)
encoded = (encoded / max_per_col)
encoded.iloc[:, label_col] = labels

train_data, test_data = split_train_test(encoded, 0.5, True)

# Remove any encoded data of the same class
train_data_features = train_data.copy()
remove_encoded_category_data_and_label(train_data_features, feature_to_encoded_cols, label_col)
train_data_features_list = train_data_features.values.tolist()

test_data_features = test_data.copy()
remove_encoded_category_data_and_label(test_data_features, feature_to_encoded_cols, label_col)
test_data_features_list = test_data_features.values.tolist()

# Data Correlation
# corr = train_data_features.corr()
# fig = px.imshow(corr, aspect="auto", origin='lower')
# fig.show()

# Get labels
train_data_labels = train_data.iloc[:, label_col].values.tolist()
test_data_labels = test_data.iloc[:,label_col].values.tolist()
train_data_features


# In[6]:


row = 0
img_data = train_data_features.iloc[row].values
classification = train_data_labels[row]
print("Classification for row " + str(row) + " is: " + str(classification))
view_image(img_data, 28, 28, 'gray')


# In[7]:


# Now train the model
hidden_layer_sizes = [10]
num_features = len(train_data_features_list[0])
num_classifications = 10
learning_rate = 1e-5
num_iterations = 5000
batch_size = 1
trained_model = rust_machine_learning.BasicNeuralNetwork(hidden_layer_sizes, num_features, num_classifications)
trained_model.train(train_data_features_list, train_data_labels, learning_rate, num_iterations, batch_size)


# In[ ]:


# Now let us test the model
labelled_results = trained_model.classify(test_data_features_list)

num_correct = 0
tolerance = 0.05
for test_label, result_label in zip(test_data_labels, labelled_results):
  num_correct += (abs(test_label-result_label)/(1.0 if test_label == 0 else test_label) <= tolerance)
print("Percent Correct: ", 100.0 * num_correct / len(test_data_labels))


# In[ ]:


# Interactive Test
import random
row = random.randint(0, len(test_data_features))
img_data = test_data_features.iloc[row].values
list_img_data = [test_data_features_list[row]]
prediction = trained_model.classify(list_img_data)[0]
print(prediction)
view_image(img_data, 28, 28, 'gray')

