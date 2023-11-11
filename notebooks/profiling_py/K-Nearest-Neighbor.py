#!/usr/bin/env python
# coding: utf-8

# In[3]:


import rust_machine_learning
import plotly.express as px
import pandas as pd
from syed_ml_lib import *


# In[4]:


# Get the csv data
df = pd.read_csv("../data/kmeans-dataset.csv")
fig = px.scatter(df, x="x", y="y")
fig.show("notebook")


# In[5]:


train_data, test_data = split_train_test(df, 0.5)


# In[5]:


x_train = train_data["x"]
y_train = train_data["y"]
labels = train_data["cluster"]
x_test = test_data["x"]
y_test = test_data["y"]
k = 200
labelled_inputs = rust_machine_learning.k_nearest_neighbor_2d(x_train, y_train, labels, x_test, y_test, k)

correct = 0
index = 0
total = len(x_test)

test_labels = test_data["cluster"]

for label in test_labels:
  if labelled_inputs[index] == label:
    correct += 1
  index+=1

print("Percent Correct: ", (correct/total))
print(len(x_train), len(x_test), len(test_labels))

