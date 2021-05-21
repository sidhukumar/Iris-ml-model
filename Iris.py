#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib


# In[12]:

os.makedirs('./outputs', exist_ok=True)
#Reading the data present in 'Iris.csv' into a dataframe - iris
iris = pd.read_csv("data/Iris.csv")


# In[13]:


#Creating a train test split. The dataset is split into a training set and a test set. In this case
#we split the data into a 70-30 split. So 70% of the data is reserved for training and 30% of the data is reserved for testing.
train, test = train_test_split(iris, test_size=0.3)
print(train.shape)
print(test.shape)

#There are always 2 dimensions X_train,Y_train and X_Test,Y_Test to build the model

#Here we train the model using 'SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm' columns.
#And we are predicting the 'Species'
X_train = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y_train = train.Species

X_test = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y_test = test.Species


# In[14]:


#Using Logistic Regression for training
model = LogisticRegression() # The model that we will use. This is present in sklearn.
model.fit(X_train, Y_train) # We fit the data(X_Train,Y_Train) into the model

#Dumping as a Pickle file which will later call for scoring the model.
filename = 'iris_model.pkl'
with open(os.path.join('./outputs/', filename), 'wb') as file:
    joblib.dump(model,filename)

#Dumping the X_test and Y_Test to use it later for scoring the model.
X_test.to_csv(r'./outputs/X_Test.csv', index=True)
Y_test.to_csv(r'./outputs/Y_Test.csv', header=True)


