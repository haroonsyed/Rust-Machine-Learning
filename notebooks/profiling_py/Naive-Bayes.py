#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rust_machine_learning
import plotly.express as px
import pandas as pd
from syed_ml_lib import *


# In[3]:


# Get the csv data
df = pd.read_csv("../data/iris.data")


# In[4]:


train_data, test_data = split_train_test(df, 0.2)
print(len(train_data))


# In[200]:


# Now train the model
train_data_features = train_data.iloc[:, 0:-1].values.tolist()
train_data_labels = pd.Categorical(train_data[" class"]).codes.tolist()

trained_model = rust_machine_learning.naive_bayes_model(train_data_features, train_data_labels)


# In[201]:


# Now let us test the model
test_data_features = test_data.iloc[:, 0:-1].values.tolist()
test_data_labels = pd.Categorical(test_data[" class"]).codes.tolist()

labelled_results = trained_model.naive_bayes_gaussian(test_data_features)

num_correct = 0
for test_label, result_label in zip(test_data_labels, labelled_results):
  num_correct += test_label==result_label
print("Percent Correct: ", num_correct / len(test_data_labels))

