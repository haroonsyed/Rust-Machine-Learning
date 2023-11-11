#!/usr/bin/env python
# coding: utf-8

# In[8]:


import rust_machine_learning
import pandas as pd
from syed_ml_lib import *


# In[9]:


# Get the csv data
df = pd.read_csv("../data/iris.data")
# df = pd.read_csv("../data/kmeans-dataset.csv")
# df = pd.read_csv("../data/winequality-red.csv")
# df = pd.read_csv("../data/digit-recognizer/train.csv")
# df = df.sample(n=10000)
df  


# In[10]:


to_encode = [4] # We won't encode the label column. The output layer will use 0-n (assuming classifications are ordered starting at 0), to determine error.
encoded, feature_to_encoded_cols = one_hot_encode(df, to_encode)
encoded


# In[11]:


# Remove unnecessary data or NAN
to_remove = []
encoded = encoded.drop(encoded.columns[to_remove], axis=1)
encoded


# In[12]:


# Prepare data for training with input rust functions expect
label_col = 2

# Normalize columns
# labels = encoded.iloc[:, label_col]
# max_per_col = encoded.abs().max().replace(0,1)
# encoded = (encoded / max_per_col)
# encoded.iloc[:, label_col] = labels

train_data, test_data = split_train_test(encoded, 0.9, True)

# Remove any encoded data of the same class
train_data_features = train_data.copy()
remove_encoded_category_data_and_label(train_data_features, feature_to_encoded_cols, label_col)

train_data_features_list = train_data_features.values.tolist()

# Data Correlation
corr = train_data_features.corr()
fig = px.imshow(corr, aspect="auto", origin='lower')
fig.show()

# Get labels
train_data_labels = train_data.iloc[:, label_col].values.tolist()
train_data_features


# In[13]:


# Now train the model
hidden_layer_sizes = [20,20]
num_features = len(train_data_features_list[0])
num_classifications = 1 # Tells the library to switch to regression mode
learning_rate = 1e-4
num_iterations = 20000
batch_size = 0
train_data_features_list
trained_model = rust_machine_learning.BasicNeuralNetwork(hidden_layer_sizes, num_features, num_classifications)
trained_model.train(train_data_features_list, train_data_labels, learning_rate, num_iterations, batch_size)


# In[14]:


# Now let us test the model
test_data_features = test_data.copy()
remove_encoded_category_data_and_label(test_data_features, feature_to_encoded_cols, label_col)
test_data_features_list = test_data_features.values.tolist()
test_data_labels = test_data[test_data.columns[label_col]].values.tolist()

labelled_results = trained_model.regression(test_data_features_list)

num_correct = 0
tolerance = 0.05
for test_label, result_label in zip(test_data_labels, labelled_results):
  num_correct += (abs(test_label-result_label)/(1.0 if test_label == 0 else test_label) <= tolerance)
print("Percent Correct: ", 100.0 * num_correct / len(test_data_labels))
labelled_results

