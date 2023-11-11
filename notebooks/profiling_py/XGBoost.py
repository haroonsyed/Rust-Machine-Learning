#!/usr/bin/env python
# coding: utf-8

# In[91]:


import rust_machine_learning
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from syed_ml_lib import *
import plotly.io as pio


# In[121]:


# Get the csv data
# df = pd.read_csv("../data/iris.data")
# df = pd.read_csv("../data/kmeans-dataset.csv")
# df = pd.read_csv("../data/winequality-red.csv")
df = pd.read_csv("../data/breast_cancer_wisconsin.csv")
df  


# In[125]:


to_encode = [1]
encoded, feature_to_encoded_cols = one_hot_encode(df, to_encode)
encoded


# In[126]:


to_remove = [0,31]
encoded = encoded.drop(encoded.columns[to_remove], axis=1)
encoded


# In[166]:


# Prepare data for training with input rust functions expect
label_col = 0

train_data, test_data = split_train_test(encoded, 0.7, True)

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


# In[167]:


# Now train the model
num_trees = 10
xgb_lambda = rust_machine_learning.XGB.default_lamba()
xgb_gamma = rust_machine_learning.XGB.default_gamma()
xgb_eta = rust_machine_learning.XGB.default_eta()
xgb_max_depth = rust_machine_learning.XGB.default_max_depth()
xgb_sample = rust_machine_learning.XGB.default_sample()
trained_model = rust_machine_learning.XGB(train_data_features_list, train_data_labels, num_trees, xgb_lambda, xgb_gamma, xgb_max_depth, xgb_sample, xgb_eta)


# In[168]:


# Now let us test the model
test_data_features = test_data.copy()
remove_encoded_category_data_and_label(test_data_features, feature_to_encoded_cols, label_col)
test_data_features = test_data_features.values.tolist()
test_data_labels = test_data[test_data.columns[label_col]].values.tolist()

labelled_results = trained_model.classify(test_data_features, xgb_eta)

num_correct = 0
tolerance = 0.1
for test_label, result_label in zip(test_data_labels, labelled_results):
  num_correct += (abs(test_label-result_label)/(1.0 if test_label == 0 else test_label) <= tolerance)
print("Percent Correct: ", num_correct / len(test_data_labels))
labelled_results

