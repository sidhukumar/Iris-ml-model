#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd 
import pickle
import joblib


# In[21]:


#Reading the pickle file
filename = './outputs/iris_model.pkl'
model = joblib.load(filename)
model


# In[22]:


#Reading the Test data
X_test = pd.read_csv("X_Test.csv",index_col=0)
Y_test_csv = pd.read_csv("Y_Test.csv",index_col=0)
Y_test = Y_test_csv.to_numpy()


# In[23]:


#Scoring the model
score = model.score(X_test, Y_test)
print("Accuracy :", score)


# In[24]:


#Printing the Predictions
prediction = model.predict(X_test)


# In[25]:


#Predictions
Predicted_Iris = X_test.assign(Predictions = prediction) 
Predicted_Iris


# In[ ]:




