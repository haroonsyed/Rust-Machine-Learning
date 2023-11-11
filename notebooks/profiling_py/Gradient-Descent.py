#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rust_machine_learning
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from syed_ml_lib import *


# In[2]:


x,y = generate_2d_data_from_function([5,2],-3, 3, 100)
original = px.scatter(x=x, y=y)
original.update_yaxes(range=[-3,3])
original.update_xaxes(range=[-3,3])
original.show()


# In[7]:


learn_rate = 0.001
max_iter = 100
intercept, slope = rust_machine_learning.gradient_descent(x,y,learn_rate,max_iter)
intercept, slope


# In[8]:


predicted_y = [(intercept + slope * x_val) for x_val in x]

# Now graph the predicted and compare to expected
original = px.scatter(x=x, y=y)
original.update_yaxes(range=[-3,3])
original.update_xaxes(range=[-3,3])
original.show()

predicted = px.scatter(x=x, y=predicted_y)
predicted.update_yaxes(range=[-3,3])
predicted.update_xaxes(range=[-3,3])
predicted.show()

