#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rust_machine_learning
import plotly.express as px
import pandas as pd
from syed_ml_lib import *


# In[86]:


# Get the csv data
df = pd.read_csv("../data/iris.data")
# df = pd.read_csv("../data/kmeans-dataset.csv")
# df = pd.read_csv("../data/winequality-red.csv")
# df = pd.DataFrame(data={
#   "chest_pain": [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
#   "blocked_arteries": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
#   "patient_weight": [205, 180, 210, 167, 156, 125, 168, 172],
#   "heart_disease": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
# })
df  


# In[87]:


to_encode = [4]
encoded, feature_to_encoded_cols = one_hot_encode(df, to_encode)
encoded


# In[88]:


# Prepare data for training with input rust functions expect
label_col = 6

train_data, test_data = split_train_test(encoded, 0.75, True)

# Remove any encoded data of the same class
train_data_features = train_data.copy()
remove_encoded_category_data_and_label(train_data_features, feature_to_encoded_cols, label_col)
train_data_features_list = train_data_features.values.tolist()

# Get labels
train_data_labels = train_data.iloc[:, label_col].values.tolist()

# Determine which columns in train_data_features_list are categorical
is_categorical = [False, False, False, False]
# is_categorical = [True, True, False]

train_data_features


# In[89]:


# Now train the model
trained_model = rust_machine_learning.AdaBoost(
    train_data_features_list, is_categorical, train_data_labels, 10
)

trained_model.print()


# In[90]:


# Now let us test the model
test_data_features = test_data.copy()
remove_encoded_category_data_and_label(test_data_features, feature_to_encoded_cols, label_col)
test_data_features = test_data_features.values.tolist()
test_data_labels = test_data[test_data.columns[label_col]].values.tolist()

labelled_results = trained_model.classify(test_data_features)

num_correct = 0
for test_label, result_label in zip(test_data_labels, labelled_results):
  num_correct += test_label==result_label
print("Percent Correct: ", num_correct / len(test_data_labels))

