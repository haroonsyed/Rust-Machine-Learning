#!/usr/bin/env python
# coding: utf-8

# In[39]:


import rust_machine_learning
import plotly.express as px
import pandas as pd


# In[40]:


# Grab the data
df = pd.read_csv('../data/kmeans-dataset.csv')
fig = px.scatter(df, x="x", y="y")
fig.show("notebook")


# In[42]:


# Now let's run our kmeans and plot the clustered results
x_values = df["x"]
y_values = df["y"]

num_clusters = 9
centers = rust_machine_learning.k_means_cluster_2d(num_clusters, x_values, y_values)
result = pd.DataFrame({"x": [], "y": [], "calculated_cluster": []})
for (x, y) in zip(x_values, y_values):
  closest_cluster = rust_machine_learning.get_closest_center_2d(centers, x, y)
  result.loc[len(result)] = ({"x": x, "y": y, "calculated_cluster": closest_cluster})

result_fig = px.scatter(result, x="x", y="y", color="calculated_cluster")
result_fig.show("notebook")

