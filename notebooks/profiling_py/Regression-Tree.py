#!/usr/bin/env python
# coding: utf-8

# In[74]:


import rust_machine_learning
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from syed_ml_lib import *


# In[75]:


# Get the csv data
# df = pd.read_csv("../data/iris.data")
df = pd.read_csv("../data/kmeans-dataset.csv")
# df = pd.read_csv("../data/winequality-red.csv")
# df = pd.DataFrame({
#   "Dosage": [0.0,2.0,4.0,8.0,11.0,13.0,15.0,18.0,20.0,22.0,25.0,26.0,27.0,28.0,29.0,31.0,34.0,36.0,37.0],
#   "Effectiveness": [0.0,0.0,0.0,0.0,5.0,18.0,100.0,100.0,100.0,100.0,60.0,58.0,56.0,52.0,48.0,15.0,0.0,0.0,0.0]
# })
df  


# In[76]:


to_encode = [2]
encoded, feature_to_encoded_cols = one_hot_encode(df, to_encode)
encoded


# In[77]:


# Now setup the train/test data
# We will train it to predict y based on x and cluster
label_col = 1

train_data, test_data = split_train_test(encoded, 0.8)

# Feature Data
train_data_features = train_data.copy()
remove_encoded_category_data_and_label(train_data_features, feature_to_encoded_cols, label_col)
train_data_features_list = train_data_features.values.tolist()

# Label Data
train_data_labels = train_data[train_data.columns[label_col]].values.tolist()
train_data_features


# In[78]:


# Now train the model
datapoints_per_node = 20
trained_model = rust_machine_learning.RegressionTree(train_data_features_list, train_data_labels, datapoints_per_node)
# trained_model.print()


# In[79]:


# Test the model
test_data_features = test_data.copy()
remove_encoded_category_data_and_label(test_data_features, feature_to_encoded_cols, label_col)
test_data_features_list = test_data_features.values.tolist()
test_data_labels = test_data[test_data.columns[label_col]].values.tolist()

labelled_results = trained_model.classify(test_data_features_list)

num_correct = 0
tolerance = 0.1
for test_label, result_label in zip(test_data_labels, labelled_results):
  num_correct += (abs(test_label-result_label)/test_label <= tolerance)
print("Percent Correct: ", num_correct / len(test_data_labels))


# In[80]:


# Graph labelled data by algorithm
result = test_data_features.copy()
result["y"] = labelled_results

fig = px.scatter(result, x="x", y="y")
fig.layout.title = "Labelled Result"
fig.show("notebook")
fig2 = px.scatter(test_data, x="x", y="y")
fig2.layout.title = "Expected Result"
fig2.show("notebook")

